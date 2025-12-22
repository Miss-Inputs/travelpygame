from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from travelpygame import tpg_api

from .classes import Round, Submission

if TYPE_CHECKING:
	from aiohttp import ClientSession


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
