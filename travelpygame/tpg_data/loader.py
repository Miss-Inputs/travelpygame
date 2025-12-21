import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from .classes import Round, round_list_adapter
from .main_tpg_import import get_main_tpg_rounds

if TYPE_CHECKING:
	from aiohttp import ClientSession



async def get_main_tpg_rounds_with_path(
	path: Path | None = None, game: int = 1, session: 'ClientSession | None' = None
) -> list[Round]:
	if path:
		try:
			content = await asyncio.to_thread(path.read_bytes)
			return round_list_adapter.validate_json(content)
		except FileNotFoundError:
			pass
	rounds = await get_main_tpg_rounds(game, session)
	if path:
		j = round_list_adapter.dump_json(rounds, indent=4, exclude_none=True)
		await asyncio.to_thread(path.write_bytes, j)
	return rounds


def load_rounds(path: Path) -> list[Round]:
	"""Loads a list of rounds from a JSON file."""
	content = path.read_bytes()
	return round_list_adapter.validate_json(content)


async def load_rounds_async(path: Path) -> list[Round]:
	"""Loads a list of rounds from a JSON file in another thread."""
	content = await asyncio.to_thread(path.read_bytes)
	return round_list_adapter.validate_json(content)
