"""API used by the new site, https://travelpicsgame.com"""

from datetime import datetime

from aiohttp import ClientSession
from pydantic import BaseModel, Field, TypeAdapter
from tqdm.auto import tqdm

user_agent = 'https://github.com/Miss-Inputs/travelpygame'


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


async def get_rounds(game: int = 1, session: ClientSession | None = None) -> list[TPGRound]:
	if session is None:
		async with get_session() as sesh:
			return await get_rounds(game, sesh)
	url = f'https://travelpicsgame.com/api/v1/rounds/{game}'
	async with session.get(url) as response:
		response.raise_for_status()
		text = await response.text()
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
	round_num: int, game: int = 1, session: ClientSession | None = None
) -> list[TPGSubmission]:
	if session is None:
		async with get_session() as sesh:
			return await get_round_submissions(round_num, game, sesh)
	url = f'https://travelpicsgame.com/api/v1/submissions/{game}/1/round/{round_num}'
	async with session.get(url) as response:
		response.raise_for_status()
		text = await response.text()
	return _sub_list_adapter.validate_json(text)


async def get_all_submissions(
	max_round_num: int | None, game: int = 1, session: ClientSession | None = None
) -> dict[int, list[TPGSubmission]]:
	"""Gets all Travel Pics Game submissions that have been added to the map.

	Arguments:
		max_round_num: Latest round number if known, this will also display the progress bar better.
		session: Optional requests.Session if you have one, otherwise creates a new one.
		timeout: Request timeout in seconds, defaults to 10 seconds.

	Returns:
		{round number: list of submissions for that round}
	"""
	# Could be parallelized, but that might not be very nice, unless it's just using a lock anyway in which case it wouldn't necessarily accomplish much
	if session is None:
		async with get_session() as sesh:
			return await get_all_submissions(max_round_num, game, sesh)

	subs: dict[int, list[TPGSubmission]] = {}
	round_num = 1
	with tqdm(total=max_round_num) as t:
		while (max_round_num is None) or (round_num <= max_round_num):
			round_subs = await get_round_submissions(round_num, game, session)
			t.update()
			if not round_subs:
				break
			subs[round_num] = round_subs
			round_num += 1
	return subs


class TPGPlayer(BaseModel):
	discord_id: str
	name: str
	username: str | None


_player_list_adapter = TypeAdapter(list[TPGPlayer])


async def get_players(session: ClientSession | None = None) -> list[TPGPlayer]:
	if session is None:
		async with get_session() as sesh:
			return await get_players(sesh)
	url = 'https://travelpicsgame.com/api/v1/players'
	async with session.get(url) as response:
		response.raise_for_status()
		text = await response.text()
	return _player_list_adapter.validate_json(text)


# TODO: /api/v1/games could be something
# TODO: /api/v1/submissions/user/{discord_id} could be something if getting an individual player
