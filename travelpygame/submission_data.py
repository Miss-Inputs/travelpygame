"""Fetches and organises data of all submissions including official and unofficial games, groups into submissions by each user, etc."""

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Self

import geopandas
from async_lru import alru_cache
from geopandas import GeoDataFrame
from pydantic_core import from_json
from shapely import Point
from tqdm.auto import tqdm

from travelpygame.tpg_data.main_tpg_import import get_player_id_names, normalize_player_name

from .point_set import PointSet
from .tpg_api import GameID, PlayerID, ServerID, get_games, get_round_submissions, get_rounds
from .tpg_api import get_session as get_official_api_session
from .util import read_geodataframe

if TYPE_CHECKING:
	from aiohttp import ClientSession

	from .tpg_data.classes import PlayerName

logger = logging.getLogger(__name__)


@dataclass
class SubmissionInfo:
	"""Holds info on a single instance of a submission."""

	player_name: 'PlayerName'
	"""Can be assumed/trusted to be a unique key"""
	player_id: PlayerID | None
	point: Point
	"""Where this submission is."""
	rounded: tuple[float, float]
	"""Rounded coordinates for `point` as (lat, lng)."""
	game_name: str
	discord_server: str
	official_game_id: GameID | None = None
	"""If from the official API, which game ID it is."""
	round_num: int | None = None
	"""If known, this is what round it was in."""
	season: int | None = None
	"""If known, this is what season the round was."""
	round_start_time: datetime | None = None
	"""If known, this is when the round started, i.e. submission was submitted after this."""


@dataclass
class RoundInfo:
	game_name: str
	season: int | None
	round_num: int | None
	target: Point
	official_game_id: int | None = None
	start_time: datetime | None = None
	"""If known"""


# These are sort of duplicating tpg_data Submission/RoundInfo, oh well, different purpose


@dataclass
class GroupedSubmission:
	"""A unique instance of a location by one player and how often it has been submitted by that player, etc.
	Probably needs a better name.
	"""

	player: 'PlayerName'
	"""Player name/username, can be assumed to be unique."""
	point: Point
	"""Where this submission is (WGS84). The exact value can be anywhere out of the instances of this submission as per rounding."""
	rounded: tuple[float, float]
	"""Rounded coordinates as (lat, lng)."""
	count: int
	"""Number of times this has been known to be submitted."""
	earliest_known: datetime | None
	"""Earliest round start time that this has been known to have been submitted after."""
	latest_known: datetime | None
	"""Latest round start time that this has been known to have been submitted during."""
	game_names: set[str]
	"""Names of all games that this has been known to be submitted to."""


def _group_player_submissions(
	player_name: 'PlayerName', sub_infos: list[SubmissionInfo]
) -> list[GroupedSubmission]:
	by_rounded_coords: defaultdict[tuple[float, float], list[SubmissionInfo]] = defaultdict(list)
	for sub_info in sub_infos:
		by_rounded_coords[sub_info.rounded].append(sub_info)

	subs = []

	for rounded_coords, sub_instances in by_rounded_coords.items():
		point = sub_instances[0].point
		# We could try and get the most detailed coordinate, but that's not really necessary
		all_dates = [sub.round_start_time for sub in sub_instances if sub.round_start_time]
		earliest = min(all_dates) if all_dates else None
		latest = max(all_dates) if all_dates else None
		game_names = {sub.game_name for sub in sub_instances}

		subs.append(
			GroupedSubmission(
				player_name, point, rounded_coords, len(sub_instances), earliest, latest, game_names
			)
		)

	return subs


def group_submissions(sub_infos: list[SubmissionInfo]) -> list[GroupedSubmission]:
	sub_info_by_player: defaultdict[PlayerName, list[SubmissionInfo]] = defaultdict(list)
	for sub_info in sub_infos:
		sub_info_by_player[sub_info.player_name].append(sub_info)

	subs = []
	for player, player_subs in sub_info_by_player.items():
		subs += _group_player_submissions(player, player_subs)
	return subs


@dataclass
class AllSubmissionData:
	"""Stores submission info and round info as separate things."""

	# Mainly the round info is just here because the Cellery tracker export has that data so why not

	submissions: list[SubmissionInfo]
	"""All submissions known to exist."""
	rounds: list[RoundInfo]
	"""All rounds known to exist."""

	@property
	def grouped_submissions(self) -> list[GroupedSubmission]:
		return group_submissions(self.submissions)

	# could have some other utility methods in this class I guess


def player_submissions_to_point_set(name: str, submissions: list[GroupedSubmission]) -> PointSet:
	"""Converts a list of one player's known submissions to a PointSet."""
	# TODO: Other fanciness, maybe do a little reverse geocoding
	gdf = GeoDataFrame(submissions, geometry='point', crs='wgs84')
	return PointSet(gdf, name)


def _group_by_player(subs: list[GroupedSubmission]) -> dict['PlayerName', list[GroupedSubmission]]:
	per_player: defaultdict[PlayerName, list[GroupedSubmission]] = defaultdict(list)
	for sub in subs:
		per_player[sub.player].append(sub)
	return per_player


def get_all_point_sets(
	submissions: list[GroupedSubmission] | dict['PlayerName', list[GroupedSubmission]],
	minimum_datetime: datetime | None = None,
	minimum_count: int | None = None,
) -> list[PointSet]:
	"""
	Converts all SubmissionPoints to point sets for each player.
	Arguments:
		submissions: List of all known unique submission points, or dict of player -> submissions per player.
		minimum_datetime: Only include players that have been known to be active since at least this date.
	"""
	by_player = _group_by_player(submissions) if isinstance(submissions, list) else submissions
	point_sets = []
	for player, subs in by_player.items():
		if minimum_datetime:
			dates = [sub.earliest_known for sub in subs if sub.earliest_known]
			if not dates or max(dates) < minimum_datetime:
				continue
		if minimum_count and len(subs) < minimum_count:
			continue
		point_sets.append(player_submissions_to_point_set(player, subs))
	return point_sets


def _deserialize_gdf(gdf: GeoDataFrame) -> list[GroupedSubmission]:
	rows = gdf.to_dict('records')
	subs = []
	for row in rows:
		# TODO: Validate everything, probably
		rounded_lat = row['rounded_lat']
		rounded_lng = row['rounded_lng']
		earliest_str = row['earliest_known']
		earliest_known = (
			datetime.fromisoformat(earliest_str) if isinstance(earliest_str, str) else None
		)
		latest_str = row['latest_known']
		latest_known = datetime.fromisoformat(latest_str) if isinstance(latest_str, str) else None
		game_names = set(from_json(row['game_names']))
		subs.append(
			GroupedSubmission(
				row['player'],
				row['geometry'],
				(rounded_lat, rounded_lng),
				row['count'],
				earliest_known,
				latest_known,
				game_names,
			)
		)
	return subs


def _deserialize_geojson(file: Path | bytes) -> list[GroupedSubmission]:
	if isinstance(file, bytes):
		gdf = geopandas.read_file(BytesIO(file), driver='GeoJSON')
	else:
		gdf = geopandas.read_file(file)
	return _deserialize_gdf(gdf)


def _serialize_geojson(subs: list[GroupedSubmission]) -> str:
	"""Serializes submissions to GeoJSON with no funny data types that don't like to go in GeoJSONs. There might be an easier way to do this, but that'll do"""
	rows = [
		{
			'player': sub.player,
			'point': sub.point,
			'rounded_lat': sub.rounded[0],
			'rounded_lng': sub.rounded[1],
			'count': sub.count,
			'earliest_known': sub.earliest_known.isoformat() if sub.earliest_known else None,
			'latest_known': sub.latest_known.isoformat() if sub.latest_known else None,
			'game_names': json.dumps(list(sub.game_names)),
		}
		for sub in subs
	]
	gdf = GeoDataFrame(rows, geometry='point', crs='wgs84')
	return gdf.to_json(indent='\t', ensure_ascii=False)


class SubmissionSummary:
	def __init__(self, submissions: list[GroupedSubmission]):
		self.submissions = submissions

	@classmethod
	def from_file(cls, path: Path) -> Self:
		subs = _deserialize_geojson(path)
		return cls(subs)

	def save_to_file(self, path: Path):
		geojson = _serialize_geojson(self.submissions)
		return path.write_text(geojson, 'utf-8')

	@cached_property
	def per_player(self) -> dict['PlayerName', list[GroupedSubmission]]:
		return _group_by_player(self.submissions)


# TODO: Function to create AllSubmissionData from tpg_data classes (list of Round etc)
# TODO: We probably want a simplified version of this stuff for get_submission_summary to avoid the intermediate step of looking at occurrence details like game name et

SERVER_NAMES: dict[ServerID, str] = {'730647011497607220': 'CG', '851583874768044052': 'US'}
"""Official TPG API /games just has the Discord server IDs, so just convert them here for consistency"""


async def get_all_official_data(
	rounding: int | None = 6,
	session: 'ClientSession | None' = None,
	*,
	forbid_extra: bool = False,
	disable_tqdm: bool = False,
) -> AllSubmissionData:
	"""Gets AllSubmissionData from official TPG API (hence, does not include unofficial spinoffs)."""
	# This is kind of duplicating a bit from main_tpg_import for now, but eh…
	if session is None:
		async with get_official_api_session() as sesh:
			return await get_all_official_data(
				rounding, sesh, forbid_extra=forbid_extra, disable_tqdm=disable_tqdm
			)

	round_infos = []
	submissions = []

	player_names = await get_player_id_names(session)

	games = await get_games(session, forbid_extra=forbid_extra)
	for game in tqdm(
		games, desc='Getting official TPG API rounds', unit='game', disable=disable_tqdm
	):
		server_name = SERVER_NAMES.get(game.server_id, f'<{game.server_id}>')
		rounds = await get_rounds(game.id, session, forbid_extra=forbid_extra)
		for r in tqdm(
			rounds,
			desc=f'Getting round submissions for {game.name}',
			unit='round',
			disable=disable_tqdm,
		):
			# Should this be rounded? Shrug
			target = Point(r.longitude, r.latitude)
			round_infos.append(
				RoundInfo(game.name, r.season, r.number, target, game.id, r.start_timestamp)
			)
			subs = await get_round_submissions(
				r.number, game.id, session, forbid_extra=forbid_extra
			)
			for sub in subs:
				player_name = player_names.get(sub.discord_id, f'<{sub.discord_id}>')
				point = Point(sub.longitude, sub.latitude)
				lat = round(sub.latitude, rounding) if rounding is not None else sub.latitude
				lng = round(sub.longitude, rounding) if rounding is not None else sub.longitude
				submissions.append(
					SubmissionInfo(
						player_name,
						sub.discord_id,
						point,
						(lat, lng),
						game.name,
						server_name,
						game.id,
						r.number,
						r.season,
						r.start_timestamp,
					)
				)

	return AllSubmissionData(submissions, round_infos)


@alru_cache
async def get_round_starts(
	game_id: GameID, session: 'ClientSession | None' = None, *, forbid_extra: bool = False
) -> dict[int, datetime]:
	rounds = await get_rounds(game_id, session, forbid_extra=forbid_extra)
	return {r.number: r.start_timestamp for r in rounds if r.start_timestamp is not None}


async def convert_cellery_geojson(  # ruff: ignore[complex-structure] #meh
	path: Path,
	rounding: int | None = 6,
	aliases: Mapping['PlayerName', 'PlayerName'] | None = None,
	tpg_api_session: 'ClientSession | None' = None,
	*,
	get_tpg_api_info: bool = True,
	forbid_extra: bool = False,
) -> AllSubmissionData:
	"""Gets AllSubmissionData from Cellery's tools site (https://tpg.odder.dev/tracker/settings)."""
	if get_tpg_api_info and tpg_api_session is None:
		async with get_official_api_session() as sesh:
			return await convert_cellery_geojson(
				path, rounding, aliases, sesh, get_tpg_api_info=True, forbid_extra=forbid_extra
			)

	aliases = aliases or {}
	gdf = read_geodataframe(path)
	rows = gdf.to_dict(orient='records')

	if get_tpg_api_info:
		official_games = await get_games(tpg_api_session, forbid_extra=forbid_extra)
		game_ids = {game.name: game.id for game in official_games}
		start_times = {
			game.id: await get_round_starts(game.id, tpg_api_session, forbid_extra=forbid_extra)
			for game in official_games
		}
	else:
		game_ids = {}
		start_times = {}

	submissions = []
	rounds = []
	for row in rows:
		# TODO: Could get player_id from TPG API, but don't really need that info for anything
		row_type = row['type']
		assert isinstance(row_type, str), f'row_type in {path} was {type(row_type)} and not str'
		point = row['geometry']
		if not isinstance(point, Point):
			raise TypeError(
				f'TPG export {path} contained a {type(point)} instead of Point: {point!r}'
			)
		lat = round(point.y, rounding) if rounding is not None else point.y
		lng = round(point.x, rounding) if rounding is not None else point.x

		game_name = row['game']
		assert isinstance(game_name, str), f'game_name in {path} was {type(game_name)} and not str'
		round_name = row['round']
		try:
			round_num = int(round_name)
		except ValueError:
			# That can happen, but we'll just ignore it
			# TODO: Do something with non-numeric rounds (e.g. "GE1" in TPG Tournament)
			round_num = None
		season_name = row['season']
		try:
			season = int(season_name)
		except ValueError:
			# TODO: Do something with non-numeric seasons (e.g. "All" in Losers TUILET)
			season = None

		if row_type == 'guess':
			server_name = row['discord_server']
			game_id = game_ids.get(game_name) if server_name == 'Official TPG API' else None
			round_start_time = None
			if game_id is not None:
				round_starts = start_times.get(game_id, {})
				round_start_time = round_starts.get(round_num)
			name = normalize_player_name(row['username'])
			name = aliases.get(name, name)
			if name == '<invalid>':
				continue

			submissions.append(
				SubmissionInfo(
					name,
					None,
					point,
					(lat, lng),
					game_name,
					server_name,
					game_id,
					round_num,
					season,
					round_start_time,
				)
			)
		elif row_type == 'answer':
			# Round target rows don't have the source/Discord server, so we can't say for sure that they are from the TPG API, so like ehhh
			rounds.append(RoundInfo(game_name, season, round_num, point))
		else:
			raise ValueError(f'Unknown row type {row_type} found in {path}')

	return AllSubmissionData(submissions, rounds)
