"""Organizes TPG submissions into dicts per player."""

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas
import pandas
from pydantic import ValidationError
from tqdm.auto import tqdm

from travelpygame import tpg_api
from travelpygame.util import load_points

from .morphior_api import get_morphior_all_submissions
from .tpg_data import (
	PlayerName,
	PlayerUsername,
	Round,
	combine_player_submissions_to_point_sets,
	get_main_tpg_rounds_with_path,
	load_rounds,
)
from .tpg_data.combine import PointCounters, combine_player_submissions, combine_point_counters

if TYPE_CHECKING:
	from aiohttp import ClientSession


async def _get_submissions_from_tpg_api(
	game_id: int, session: 'ClientSession', client_timeout: float | None = 60.0
):
	per_user: defaultdict[tpg_api.PlayerID, list[tuple[float, float]]] = defaultdict(list)
	rounds = await tpg_api.get_rounds(game_id, session, client_timeout)
	for roundy in tqdm(rounds, f'Getting submissions for game ID {game_id}', unit='round'):
		subs = await tpg_api.get_round_submissions(roundy.number, game_id, session, client_timeout)
		for sub in subs:
			per_user[sub.discord_id].append((sub.latitude, sub.longitude))
	return per_user


async def get_main_tpg_subs_per_player(
	rounding: int = 6,
	aliases: Mapping[PlayerName, PlayerUsername] | None = None,
	session: 'ClientSession|None' = None,
	client_timeout: float | None = 60.0,
) -> PointCounters:
	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_main_tpg_subs_per_player(rounding, aliases, sesh, client_timeout)

	players = await tpg_api.get_players(session, client_timeout)
	player_names = {player.discord_id: player.username or player.name for player in players}
	games = await tpg_api.get_games(session, client_timeout)
	combined: defaultdict[tpg_api.PlayerID, list[tuple[float, float]]] = defaultdict(list)
	for game in games:
		game_subs = await _get_submissions_from_tpg_api(game.id, session, client_timeout)
		for player_id, subs in game_subs.items():
			player_name = player_names.get(player_id, f'<unknown {player_id}>')
			combined[player_name] += subs

	return combine_player_submissions((combined,), aliases, rounding)


async def get_morphior_subs_per_player(
	rounding: int = 6,
	aliases: Mapping[PlayerName, PlayerUsername] | None = None,
	session: 'ClientSession | None' = None,
):
	gdf = await get_morphior_all_submissions(session)
	rows = ((row['username'], row.geometry.y, row.geometry.x) for _, row in gdf.iterrows())
	return combine_player_submissions((rows,), aliases, rounding)


def get_per_player_subs_from_gdf(gdf: geopandas.GeoDataFrame):
	subs: dict[PlayerUsername, geopandas.GeoDataFrame] = {}
	for player, group in gdf.groupby('player', sort=False):
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
		subs[str(player)] = group.rename_axis(index=str(player))
	return subs


def load_per_player_submissions(path: Path):
	"""Loads submissions per player from a file. If path is a geofile, it _must_ have a column named "player" with the player name of each point, and if it has a column named "name" that is unique for each player then that will be used as the name of each pic (the index in the GeoSeries). Otherwise, it will load a list of rounds from that file."""
	if path.suffix.lower() == '.json':
		try:
			rounds = load_rounds(path)
		except ValidationError:
			pass
		else:
			# Combines without aliases and default rounding but meh
			return combine_player_submissions_to_point_sets(rounds, None)
		# TODO: Load Season as well as list[Round]
	points = load_points(path)
	return get_per_player_subs_from_gdf(points)


def save_submissions_per_user(subs: Mapping[PlayerUsername, geopandas.GeoDataFrame], path: Path):
	all_subs = pandas.concat(
		(points.assign(player=player) for player, points in subs.items()), ignore_index=True
	)
	assert isinstance(all_subs, geopandas.GeoDataFrame)
	all_subs.to_file(path)


async def load_or_fetch_per_player_submissions(
	path: Path | None = None,
	main_data_path: Path | list[Round] | None = None,
	aliases: Mapping[PlayerName, PlayerUsername] | None = None,
	rounding: int = 6,
	session: 'ClientSession | None' = None,
	*,
	load_main_data: bool = True,
	load_morphior_data: bool = True,
) -> dict[PlayerUsername, geopandas.GeoDataFrame]:
	"""Loads per-player submission data from a file, or fetches it if it is not there.

	Arguments:
		load_main_data: Fetch data from main TPG if we do not have it (will still load it from the path if it is there.)"""
	if path:
		try:
			return await asyncio.to_thread(load_per_player_submissions, path)
		except FileNotFoundError:
			pass
	if main_data_path:
		# TODO: This should load a Season when we store that instead of list[Round]
		# Well, it should also try and load all seasons of all games, but we're getting there
		# Also have the option for it to be already loaded, I guess
		main_data = (
			await get_main_tpg_rounds_with_path(main_data_path, session=session)
			if isinstance(main_data_path, Path)
			else main_data_path
		)
		main_data_counts = combine_player_submissions(main_data, aliases, rounding)
	elif load_main_data:
		main_data_counts = await get_main_tpg_subs_per_player(rounding, aliases, session)
	else:
		main_data_counts = {}

	if load_morphior_data:
		morphior_counts = await get_morphior_subs_per_player(rounding, aliases, session)
	else:
		morphior_counts = {}

	per_player = combine_point_counters((main_data_counts, morphior_counts))
	if per_player and path:
		await asyncio.to_thread(save_submissions_per_user, per_player, path)
	return per_player
