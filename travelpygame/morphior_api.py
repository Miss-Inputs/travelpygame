"""Functions to access Morphior's site to get all.geojson, because it seemed best to put them in a different module."""

from typing import Annotated, Literal

from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter
from tqdm.auto import tqdm

from .util.web import get_bytes_streamed, get_text, user_agent

TrackerID = int
UnofficialGameID = int


async def get_all_players_json(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	"""Returns json as plain text. Separate function from get_all_players in case you want to parse it in some custom way."""
	# Maybe that's not necessary to do that, though, and I should just make one function… hrm
	url = 'https://tpg.marsmathis.com/api/players'
	return await get_text(url, None, session, client_timeout)


class MorphiorPlayer(BaseModel):
	name: str
	canonical_name: str
	discord_id: str
	aliases: list[str]


_player_list_adapter = TypeAdapter(list[MorphiorPlayer])


async def get_all_players(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	json_text = await get_all_players_json(session, client_timeout)
	return _player_list_adapter.validate_json(json_text)


async def get_all_submissions_json(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	"""Returns json as undecoded bytes. Separate function from get_all_submissions in case you want to parse it in some custom way."""
	# Maybe this should be paginated with limit/offset params…
	# TODO: Other params (maybe as separate function): lat_min, lat_max, lon_min, lon_max, count_min, count_max; maybe belongs in a different function (as stream=true may be unnecessary)
	url = 'https://tpg.marsmathis.com/api/submissions'
	params = {'stream': 'true'}
	return await get_bytes_streamed(url, params, session, client_timeout)


class OfficialSubmissionOccurrence(BaseModel):
	type: Literal['official']
	game_id: int
	"""ID of which official game this is, as used in official TPG API"""
	round: int
	"""Round number within the game"""


class UnofficialSubmissionOccurrence(BaseModel):
	type: Literal['unofficial']
	tracker_id: TrackerID
	unofficial_game_id: UnofficialGameID | None
	layer: str | None


SubmissionOccurrence = Annotated[
	OfficialSubmissionOccurrence | UnofficialSubmissionOccurrence, Field(discriminator='type')
]


class MorphiorSubmission(BaseModel):
	discord_id: str
	lat: float
	lon: float
	count: int
	"""Number of times this has been submitted"""
	occurrences: list[SubmissionOccurrence]


_submission_list_adapter = TypeAdapter(list[MorphiorSubmission])


async def get_all_submissions(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
) -> list[MorphiorSubmission]:
	if session is None:
		async with ClientSession(headers={'User-Agent': user_agent}) as sesh:
			return await get_all_submissions(sesh, client_timeout or 60)

	timeout = (
		ClientTimeout(client_timeout)
		if isinstance(client_timeout, (float, int))
		else client_timeout
	)
	url = 'https://tpg.marsmathis.com/api/submissions'
	params = {'stream': 'true'}

	submissions = []
	async with session.get(url, params=params, timeout=timeout, raise_for_status=True) as response:
		with tqdm(desc='Getting all submissions', unit='submission') as t:
			while True:
				line = await response.content.readline()
				if not line:
					break
				t.update()
				submission = MorphiorSubmission.model_validate_json(line)
				submissions.append(submission)
	return submissions


async def get_player_submissions_json(
	discord_id: str,
	session: 'ClientSession | None' = None,
	client_timeout: 'float | ClientTimeout | None' = 60,
):
	# TODO: lon/lat/count max/min parameters
	url = f'https://tpg.marsmathis.com/api/submissions/{discord_id}'
	return await get_text(url, None, session, client_timeout)


async def get_player_submissions(
	discord_id: str,
	session: 'ClientSession | None' = None,
	client_timeout: 'float | ClientTimeout | None' = 60,
):
	json_text = await get_player_submissions_json(discord_id, session, client_timeout)
	return _submission_list_adapter.validate_json(json_text)


async def get_unofficial_games_json(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	url = 'https://tpg.marsmathis.com/api/games/unofficial'
	return await get_text(url, None, session, client_timeout)


class UnofficialGame(BaseModel):
	id: UnofficialGameID
	name: str
	discord_server: str
	"""CG, US, AU, etc"""
	season_start: int


_unofficial_game_adapter = TypeAdapter(list[UnofficialGame])


async def get_unofficial_games(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	json_text = await get_unofficial_games_json(session, client_timeout)
	return _unofficial_game_adapter.validate_json(json_text)


class Tracker(BaseModel):
	tracker_id: TrackerID
	name: str
	game_name: str
	discord_server: str
	season: int
	unofficial_game_id: UnofficialGameID
	url: HttpUrl


_tracker_list_adapter = TypeAdapter(list[Tracker])


async def get_unofficial_game_trackers(
	unofficial_game_id: UnofficialGameID,
	session: 'ClientSession | None' = None,
	client_timeout: 'float | ClientTimeout | None' = 60,
):
	url = f'https://tpg.marsmathis.com/api/games/unofficial/{unofficial_game_id}/trackers'
	json_text = await get_text(url, None, session, client_timeout)
	return _tracker_list_adapter.validate_json(json_text)


async def get_all_trackers(
	discord_server: str | None,
	session: 'ClientSession | None' = None,
	client_timeout: 'float | ClientTimeout | None' = 60,
):
	url = 'https://tpg.marsmathis.com/api/trackers'
	params = {'discord_server': discord_server} if discord_server else None
	json_text = await get_text(url, params, session, client_timeout)
	return _tracker_list_adapter.validate_json(json_text)
