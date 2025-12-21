import asyncio
import re
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
		s = rounds_to_json(rounds)
		await asyncio.to_thread(path.write_text, s, encoding='utf-8')
	return rounds


def load_rounds(path: Path) -> list[Round]:
	"""Loads a list of rounds from a JSON file."""
	content = path.read_bytes()
	return round_list_adapter.validate_json(content)


async def load_rounds_async(path: Path) -> list[Round]:
	"""Loads a list of rounds from a JSON file in another thread."""
	content = await asyncio.to_thread(path.read_bytes)
	return round_list_adapter.validate_json(content)


def _spaces_to_tabs(m: re.Match[str]):
	"""Whoa look out controversial opinion coming through"""
	return '\n' + m[1].replace('  ', '\t')


def rounds_to_json(rounds: list[Round]) -> str:
	"""Converts a list of rounds to nicely formatted JSON."""
	json_bytes = round_list_adapter.dump_json(rounds, indent=2, exclude_none=True)
	return re.sub(r'\n(\s{2,})', _spaces_to_tabs, json_bytes.decode('utf-8'))
