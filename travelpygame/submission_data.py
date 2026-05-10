"""Fetches and organises data of all submissions including official and unofficial games, groups into submissions by each user, etc."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas
from aiohttp import ClientSession
from shapely import Point
from tqdm.auto import tqdm

from travelpygame import load_points_async

from .morphior_api import (
	OfficialSubmissionOccurrence,
	TrackerID,
	UnofficialGameID,
	get_all_players,
	get_all_trackers,
	get_unofficial_games,
	iter_all_submissions,
)
from .point_set import PointSet
from .tpg_api import GameID, PlayerID, get_games
from .util import output_geodataframe
from .util.web import user_agent

if TYPE_CHECKING:
	from .tpg_data import PlayerName, PlayerUsername

logger = logging.getLogger(__name__)


@dataclass
class SubmissionInfo:
	"""Holds info on a single instance of a submission. Basically a flattened combination of MorphiorPlayer/MorphiorSubmission/*SubmissionOccurrence."""

	player_name: 'PlayerName | None'
	player_username: 'PlayerUsername'
	"""canonical_name from MorphiorPlayer, which is generally the username, but in the case of deleted accounts etc (which have no username in the main TPG API) this can be equal to the display name. Can be trusted to be a unique key"""
	player_id: PlayerID | None
	point: Point
	rounded: tuple[float, float]
	game_name: str
	official_game_id: GameID | None = None
	round_num: int | None = None
	"""If official, this is what round it was in"""
	# Could put tracker name or layer name in here, but for now that will do


def _default_session():
	# Same as tpg_api.get_session but eh, just in case I randomly decide to make it different in the future
	return ClientSession(headers={'User-Agent': user_agent})


async def get_game_names(
	session: ClientSession | None = None, *, forbid_extra: bool = False
) -> tuple[dict[GameID, str], dict[UnofficialGameID, str], dict[TrackerID, str]]:
	"""Returns mapping of tracker ID -> game name as well, since unofficial_game_id in UnofficialSubmissionOccurrence can be None."""
	if session is None:
		async with _default_session() as sesh:
			return await get_game_names(sesh, forbid_extra=forbid_extra)

	official = await get_games(session, forbid_extra=forbid_extra)
	unofficial = await get_unofficial_games(session, forbid_extra=forbid_extra)
	trackers = await get_all_trackers(None, session, forbid_extra=forbid_extra)

	official_names = {game.id: game.name for game in official}
	# For unofficial games we are going to go through the CGcord ones first and disambiguate anything from other servers
	spinoff_names = {game.id: game.name for game in unofficial if game.discord_server == 'CG'}
	tracker_names = {
		tracker.tracker_id: tracker.game_name
		for tracker in trackers
		if tracker.discord_server == 'CG'
	}

	for spinoff in unofficial:
		if spinoff.discord_server == 'CG':
			# already did you
			continue
		name = spinoff.name
		if name in spinoff_names.values():
			name = f'{name} ({spinoff.discord_server})'
		spinoff_names[spinoff.id] = name
	for tracker in trackers:
		if tracker.discord_server == 'CG':
			continue
		name = tracker.game_name
		if name in tracker_names.values():
			name = f'{name} ({tracker.discord_server})'
		tracker_names[tracker.id] = name

	return official_names, spinoff_names, tracker_names


async def get_submission_occurrences(
	session: ClientSession | None = None, rounding: int | None = 6, *, forbid_extra: bool = False
) -> list[SubmissionInfo]:
	"""Returns all occurrences of all submissions.
	Arguments:
		session: aiohttp session, creates a new one if None.
		rounding: Round coordinates to this amount of decimal places (or leave the coordinates exactly as is if None) for the `rounded` field, as the submissions data may have duplicate submissions that are counted as separate because they differ in precision. Note that the default is 6 digits as 1-e7 decimal degrees is around 1 or two centimetres (depending on the axis etc) and so is unlikely to matter for this use case, and 

	Returns:
		List of SubmissionInfo
	"""
	if session is None:
		async with _default_session() as sesh:
			return await get_submission_occurrences(sesh, rounding, forbid_extra=forbid_extra)

	official_names, spinoff_names, tracker_names = await get_game_names(
		session, forbid_extra=forbid_extra
	)
	players = await get_all_players(session, forbid_extra=forbid_extra)
	players_by_id = {player.discord_id: player for player in players}

	subs: list[SubmissionInfo] = []
	with tqdm(desc='Getting all submissions', unit='submission') as t:
		async for sub in iter_all_submissions(session, forbid_extra=forbid_extra):
			t.update()
			player = players_by_id.get(sub.player)
			if not player:
				logger.warning(
					'Player %s did not exist, which is strange, the submission at %s, %s will be ignored',
					sub.player,
					sub.lat,
					sub.lon,
				)
				continue
			point = Point(sub.lon, sub.lat)
			lat = round(sub.lat, rounding) if rounding is not None else sub.lat
			lon = round(sub.lon, rounding) if rounding is not None else sub.lon
			for occ in sub.occurrences:
				if isinstance(occ, OfficialSubmissionOccurrence):
					game_name = official_names.get(occ.game_id, f'<unknown game {occ.game_id}>')
					subs.append(
						SubmissionInfo(
							player.name,
							player.canonical_name,
							player.discord_id,
							point,
							(lat, lon),
							game_name,
							occ.game_id,
							occ.round,
						)
					)
				else:
					if occ.unofficial_game_id:
						game_name = spinoff_names.get(
							occ.unofficial_game_id, f'<unknown spinoff {occ.unofficial_game_id}>'
						)
					else:
						game_name = tracker_names.get(
							occ.tracker_id, f'<unknown tracker {occ.tracker_id}>'
						)
					subs.append(
						SubmissionInfo(
							player.name,
							player.canonical_name,
							player.discord_id,
							point,
							(lat, lon),
							game_name,
						)
					)

	return subs


# TODO: We probably want a simplified version of this for get_submission_summary to avoid the intermediate step of looking at occurrence details like game name etc


async def get_submission_summary(
	session: ClientSession | None = None, rounding: int | None = 6, *, forbid_extra: bool = False
):
	"""Gets all points that have been submitted somewhere at some point, and the player name and count of occurrences, etc."""
	sub_occurrences = await get_submission_occurrences(session, rounding, forbid_extra=forbid_extra)
	gdf = geopandas.GeoDataFrame(sub_occurrences, geometry='point', crs='wgs84')

	rows = []
	for player, player_group in gdf.groupby('player_username', sort=False):
		# We have to group by player name anyway since it doesn't make sense to group together different people's submissions of the same place
		for _, group in player_group.groupby('rounded', sort=False):
			first = group.iloc[0]
			row = {
				'username': player,
				'player_name': first['player_name'],
				'player_id': first['player_id'],
				'count': group.index.size,
				'geometry': first.geometry,
			}
			rows.append(row)

	return geopandas.GeoDataFrame(rows, crs='wgs84')


# TODO: We may end up wanting a get_submission_detailed_summary that aggregates things like the list of spinoffs a point has been submitted to, the first main round, etc, maybe little a reverse geocode as a treat, or to put some of those details in the current summary


async def load_or_fetch_submission_summary(
	path: Path | None = None,
	session: ClientSession | None = None,
	rounding: int | None = 6,
	*,
	forbid_extra: bool = False,
	error_if_not_found: bool = False,
):
	"""Loads the previously saved output from `get_submission_summary` if the path is provided and exists, or fetches it if not."""
	if path:
		try:
			return await load_points_async(path)
		except FileNotFoundError:
			if error_if_not_found:
				raise

	summary = await get_submission_summary(session, rounding, forbid_extra=forbid_extra)
	if path:
		await asyncio.to_thread(output_geodataframe, summary, path)
	return path


def get_all_point_sets(
	submission_summary: geopandas.GeoDataFrame, player_col_name: str | int = 'username'
):
	point_sets: list[PointSet] = []
	for name, group in submission_summary.groupby(player_col_name):
		# TODO: Option to set an index col (for the name/description of each point), but we don't have that info yet, it would just be if a custom submission_summary is provided
		name = str(name)
		data = group.drop(columns=[player_col_name, 'player_name', 'player_id'], errors='ignore')
		data = data.rename_axis(index=name)
		assert isinstance(data, geopandas.GeoDataFrame), (
			f'data was {type(data)}, expected GeoDataFrame'
		)
		point_sets.append(PointSet(data, name))
	return point_sets
