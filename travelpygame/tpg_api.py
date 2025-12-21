"""API used by the new site, https://travelpicsgame.com"""

from datetime import datetime

from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel, Field, TypeAdapter

from .util.web import get_text, user_agent


class TPGRound(BaseModel, extra='forbid'):
	number: int
	"""Round number, starting with 1 and incrementing constantly."""
	latitude: float
	longitude: float
	water: bool
	ongoing: bool
	country: str | None
	"""Two letter uppercase ISO 3166-1 code, or None if the round is not in any particular country."""
	# These two fields are strings containing ints but we automagically convert them to dates
	start_timestamp: datetime | None
	end_timestamp: datetime | None
	season: int
	game: int
	"""Which game this round is for (1 = main TPG, etc)"""


_round_list_adapter = TypeAdapter(list[TPGRound])


def get_session():
	return ClientSession(headers={'User-Agent': user_agent})


async def get_rounds(
	game: int = 1,
	session: ClientSession | None = None,
	client_timeout: ClientTimeout | float | None = 60.0,
) -> list[TPGRound]:
	url = f'https://travelpicsgame.com/api/v1/rounds/{game}'
	text = await get_text(url, session, client_timeout)
	return _round_list_adapter.validate_json(text)


class TPGSubmission(BaseModel):
	id: int
	"""Unknown, maybe an ID for each submission"""
	round: int
	"""Round number (perhaps this model is used in other places other than submissions per round?)"""
	latitude: float
	"""WGS84 latitude of the picture."""
	longitude: float
	"""WGS84 longitude of the picture."""
	place: int
	"""Placement of this picture so far, starting at 1st place, or 0th place if the round is not finished yet."""
	fivek: bool = Field(validation_alias='5k')
	"""Whether this picture counted as a 5K or not."""
	antipode_5k: bool
	discord_id: str
	"""In theory this is an int, but since it is an opaque ID, might as well leave it as str"""
	is_tie: bool
	game: int


_sub_list_adapter = TypeAdapter(list[TPGSubmission])


async def get_round_submissions(
	round_num: int,
	game: int = 1,
	session: ClientSession | None = None,
	client_timeout: ClientTimeout | float | None = 60.0,
) -> list[TPGSubmission]:
	url = f'https://travelpicsgame.com/api/v1/submissions/game/{game}/round/{round_num}'
	text = await get_text(url, session, client_timeout)
	return _sub_list_adapter.validate_json(text)


class TPGPlayer(BaseModel):
	discord_id: str
	name: str
	username: str | None


_player_list_adapter = TypeAdapter(list[TPGPlayer])


async def get_players(
	session: ClientSession | None = None, client_timeout: ClientTimeout | float | None = 60.0
) -> list[TPGPlayer]:
	"""Gets all players who have submitted for TPG."""
	url = 'https://travelpicsgame.com/api/v1/players'
	text = await get_text(url, session, client_timeout)
	return _player_list_adapter.validate_json(text)


class TPGGame(BaseModel, extra='forbid'):
	id: int
	name: str
	server_id: str
	submissions_channel: str
	scoring_function: None
	"""Unsure what this is / if it is used yet"""
	acceptance_text: None
	"""Unsure what this is / if it is used yet"""
	rejection_text: None
	"""Unsure what this is / if it is used yet"""


_games_list_adapter = TypeAdapter(list[TPGGame])


async def get_games(
	session: ClientSession | None = None, client_timeout: ClientTimeout | float | None = 60.0
) -> list[TPGGame]:
	"""Gets all games on the main site."""
	url = 'https://travelpicsgame.com/api/v1/games'
	text = await get_text(url, session, client_timeout)
	return _games_list_adapter.validate_json(text)


# /api/v1/submissions/user/{discord_id} and /api/v1/submissions/user/{discord_id}/game/{game_id} could be something if getting an individual player, otherwise, it would probably be slower and requestier to call that for every player vs. just getting all rounds
