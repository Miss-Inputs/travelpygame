import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from travelpygame import tpg_api

from .classes import PlayerName, PlayerUsername, Round, Submission

if TYPE_CHECKING:
	from aiohttp import ClientSession

logger = logging.getLogger(__name__)


def _convert_submission(
	sub: tpg_api.TPGSubmission, players: dict[str, tpg_api.TPGPlayer]
) -> Submission:
	extra = {'id': sub.id, 'discord_id': sub.discord_id, 'game': sub.game}
	if sub.discord_id in players:
		player = players[sub.discord_id]
		name = player.name
		username = player.username
	else:
		# That will have to do
		name = sub.discord_id
		username = f'<{sub.discord_id}>'
	return Submission(
		name=name,
		latitude=sub.latitude,
		longitude=sub.longitude,
		is_5k=sub.is_5k,
		is_antipode_5k=sub.antipode_5k,
		is_tie=sub.is_tie,
		username=username,
		**extra,
	)


async def get_main_tpg_rounds(game: int = 1, session: 'ClientSession | None' = None) -> list[Round]:
	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_main_tpg_rounds(game, sesh)

	api_rounds = await tpg_api.get_rounds(game, session)
	players = {player.discord_id: player for player in await tpg_api.get_players(session)}

	rounds: list[Round] = []

	with tqdm(api_rounds, 'Getting submissions', unit='round') as t:
		for round_ in t:
			api_subs = await tpg_api.get_round_submissions(round_.number, game, session)
			subs = [_convert_submission(sub, players) for sub in api_subs]
			name = f'R{round_.number}: {round_.country}' if round_.country else f'R{round_.number}'
			if round_.water:
				name += ' (water)'
			extra: dict[str, Any] = {'is_water': round_.water, 'game': round_.game}
			if round_.start_timestamp:
				extra['start_date'] = round_.start_timestamp
			if round_.end_timestamp:
				extra['end_date'] = round_.end_timestamp
			rounds.append(
				Round(
					name=name,
					number=round_.number,
					season=round_.season,
					country_code=round_.country,
					latitude=round_.latitude,
					longitude=round_.longitude,
					submissions=subs,
					**extra,
				)
			)

	return rounds


_emoji_regex_parts = (
	# This comment is so it lines up nicely when autoformatted
	r'\uE000-\uF8FF',
	r'\U0001F100-\U0001F2FF',
	r'\U0001F300-\U0001F6FF',
	r'\U0001F780-\U0001F9FF',
	r'\U0001FA70-\U0001FAFF',
	r'\U000E0000-\U000E007F',
)
_emoji_regex_middle = ''.join(_emoji_regex_parts)
probably_emoji = re.compile(rf'[{_emoji_regex_middle}]+')
"""Emoji regex kinda, excludes code blocks:
	E000 Private Use Area
	1F100 Enclosed Alphanumeric Supplement
	1F200 Enclosed Ideographic Supplement
	1F300 Misc Symbols and Pictographs
	1F600 Emoticons
	1F650 Ornamental Dingbats
	1F680 Transport and Map Symbols
	1F780 Geometric Shapes Extended
	1F900 Supplemental Symbols and Pictographs
	1FA70 Symbols and Pictographs Extended-A
	E0000 Tags
Unashamedly has false positives and false negatives, as this will never realistically be perfect unless I import a library just to do this which seems a bit much.
"""


async def get_player_username(
	name: str, session: 'ClientSession|None' = None, *, ignore_emojis: bool | None = None
) -> str | None:
	"""Finds the username of a player, from main TPG data, or None if not found.

	Arguments:
		name: Display name.
		session: Optional aiohttp session to use, will create one if not provided.
		ignore_emojis: Whether to ignore emojis (or characters that are most likely emojis, because I don't want to specify every single code block) in names or not when considering a match, by default (or if None), will be True if `name` does not include emojis or False if it does.
	"""
	if ignore_emojis is None:
		ignore_emojis = probably_emoji.search(name) is None

	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_player_username(name, sesh, ignore_emojis=ignore_emojis)

	players = await tpg_api.get_players(session)
	for player in players:
		player_name = player.name
		if ignore_emojis:
			player_name = probably_emoji.sub('', player_name)
		player_name = player_name.strip()
		if name == player_name:
			return player.username

	return None


async def get_player_display_names(
	session: 'ClientSession|None' = None, *, avoid_duplicates: bool = True
) -> dict[PlayerUsername, PlayerName]:
	"""Returns a dict mapping Discord usernames to display names.

	Arguments:
		avoid_duplicates: If true (default), maps any usernames that would cause duplicates to themselves instead of the display name.
	"""
	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_player_display_names(sesh)

	names: dict[PlayerUsername, PlayerName] = {}
	players = await tpg_api.get_players(session)
	for player in players:
		username = player.username
		if not username:
			if not player.discord_id:
				continue
			username = f'<{player.discord_id}>'
		names[username] = player.name

	if avoid_duplicates:
		counter = Counter(names.values())
		duplicate_names = {name for name, count in counter.items() if count > 1}
		if duplicate_names:
			logger.info('Duplicate display names were found: %s', duplicate_names)
			names = {
				username: username if name in duplicate_names else name
				for username, name in names.items()
			}

	return names
