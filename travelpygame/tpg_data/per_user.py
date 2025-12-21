import asyncio
import contextlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas
import pandas
import shapely
from shapely import Point
from tqdm.auto import tqdm

from travelpygame import tpg_api
from travelpygame.util import load_points

from .io import load_rounds

if TYPE_CHECKING:
	from aiohttp import ClientSession

	from .classes import Round


async def get_main_tpg_submissions_per_user(
	rounding: int = 6, session: 'ClientSession | None' = None
) -> dict[str, geopandas.GeoSeries]:
	"""Fetches submissions per-user from the API.

	Returns:
		dict: {player name: point set}"""
	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_main_tpg_submissions_per_user(rounding, sesh)
	per_user_subs: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)

	with tqdm(desc='Getting submissions', unit='submission') as t:
		games = await tpg_api.get_games(session)
		for game in games:
			rounds = await tpg_api.get_rounds(game.id, session)
			for r in rounds:
				subs = await tpg_api.get_round_submissions(r.number, game.id, session)
				for sub in subs:
					t.update()
					per_user_subs[sub.discord_id].append((sub.latitude, sub.longitude))

	player_names = {player.discord_id: player.name for player in await tpg_api.get_players(session)}
	out: dict[str, geopandas.GeoSeries] = {}
	for player_id, latlngs in per_user_subs.items():
		# Avoid duplicating points that are only off in the 7th decimal place or whatever
		points: dict[tuple[float, float], Point] = {}
		for lat, lng in latlngs:
			points[round(lat, rounding), round(lng, rounding)] = Point(lng, lat)
		player_name = player_names.get(player_id, player_id)
		out[player_name] = geopandas.GeoSeries(list(points.values()), crs='wgs84')
	return out


def get_submissions_per_user_from_path(path: Path):
	"""Gets submissions per user either from a JSON file containing TPG data, or from a geofile containing submissions with a "player" column (and optionally a "name" column for the picture description.)"""
	subs: dict[str, geopandas.GeoSeries] = {}
	if path.suffix[1:].lower() == 'json':
		rounds = load_rounds(path)
		for player, latlngs in get_submissions_per_user(rounds).items():
			player_points = shapely.points([(lng, lat) for lat, lng in latlngs]).tolist()
			assert not isinstance(player_points, Point)
			subs[player] = geopandas.GeoSeries(player_points, crs='wgs84', name=player)
	else:
		points = load_points(path)
		for player, group in points.groupby('player', sort=False):
			name_col = group.get('name')
			if name_col is not None and name_col.is_unique and not name_col.hasnans:
				with contextlib.suppress(ValueError):
					group = group.set_index('name', verify_integrity=True)
			else:
				group = group.reset_index()
			if not isinstance(group, geopandas.GeoDataFrame):
				raise TypeError(
					f'Encountered {type(group)} when grouping instead of GeoDataFrame for {player}'
				)
			subs[str(player)] = group.geometry.rename(str(player))
	return subs


async def get_submissions_per_user_with_path(
	path: Path | None = None, session: 'ClientSession | None' = None
) -> dict[str, geopandas.GeoSeries]:
	"""If path is a geofile, it _must_ have a column named "player" with the player name of each point, and if it has a column named "name" that is unique for each player then that will be used as the name of each pic (the index in the GeoSeries). Otherwise, it will load TPG data from that file. If path does not exist, it will fetch data from the API and save it there."""
	if path:
		try:
			return await asyncio.to_thread(get_submissions_per_user_from_path, path)
		except FileNotFoundError:
			pass

	subs = await get_main_tpg_submissions_per_user(session=session)
	if path and path.suffix[1:].lower() != 'json':
		all_subs = pandas.concat(
			(
				geopandas.GeoDataFrame(
					pandas.Series(player, points.index, name='player'), geometry=points
				)
				for player, points in subs.items()
			),
			ignore_index=True,
		)
		assert isinstance(all_subs, geopandas.GeoDataFrame)
		await asyncio.to_thread(all_subs.to_file, path)
	return subs


def get_submissions_per_user(rounds: Iterable['Round']):
	"""Returns dict of player name -> set[(lat, lng)] from TPG data. Probably doesn't need to be used anymore."""
	# TODO: Redo this so we can more properly round coordinates, and potentially get sub.description as the name (but it would have to be filled in with format_point as a placeholder if returning a GeoSeries… hrm)
	submissions: defaultdict[str, set[tuple[float, float]]] = defaultdict(set)

	for r in rounds:
		for sub in r.submissions:
			submissions[sub.name].add((sub.latitude, sub.longitude))
	return submissions
