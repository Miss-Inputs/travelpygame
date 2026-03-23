"""Functions to access Morphior's site to get all.geojson, because it seemed best to put them in a different module."""

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .util.web import get_bytes_streamed, get_text

if TYPE_CHECKING:
	from aiohttp import ClientSession, ClientTimeout


async def get_all_players_json(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	"""Returns json as plain text. Separate function from get_all_players in case you want to parse it in some custom way."""
	url = 'https://tpg.marsmathis.com/api/players'
	return await get_text(url, session, client_timeout)


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
	tracker_id: int
	unofficial_game_id: int | None
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


async def get_all_submissions(
	session: 'ClientSession | None' = None, client_timeout: 'float | ClientTimeout | None' = 60
):
	# This maybe wants an iterator to yield rows from ndjson instead
	json_bytes = await get_all_submissions_json(session, client_timeout)
	return [MorphiorSubmission.model_validate_json(line) for line in json_bytes.splitlines()]
