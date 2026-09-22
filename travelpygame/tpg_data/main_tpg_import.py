import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from travelpygame import tpg_api
from travelpygame.util.text import normalize_player_name

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
		**extra,  # ty:ignore[invalid-argument-type]
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


async def get_player_id_names(
	session: 'ClientSession|None' = None, *, normalize: bool = True
) -> dict[tpg_api.PlayerID, PlayerName]:
	"""Returns a dict mapping Discord IDs to display names. Any player returned by the API without a Discord ID is ignored. If any display name is duplicated, returns the username for any player with that display name instead."""
	if session is None:
		async with tpg_api.get_session() as sesh:
			return await get_player_id_names(sesh, normalize=normalize)

	names_and_usernames: dict[tpg_api.PlayerID, tuple[PlayerName, PlayerUsername]] = {}
	players = await tpg_api.get_players(session)
	for player in players:
		discord_id = player.discord_id
		if not discord_id:
			continue
		name = player.name
		if normalize:
			name = normalize_player_name(name)
		names_and_usernames[discord_id] = (name, player.username or f'{player.discord_id} {name}')

	counter = Counter(name[0] for name in names_and_usernames.values())
	duplicate_names = {name for name, count in counter.items() if count > 1}
	return {
		discord_id: username if name in duplicate_names else name
		for discord_id, (name, username) in names_and_usernames.items()
	}
