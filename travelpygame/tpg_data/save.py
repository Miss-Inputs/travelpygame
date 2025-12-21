import re

from .classes import Round, round_list_adapter


def _spaces_to_tabs(m: re.Match[str]):
	"""Whoa look out controversial opinion coming through"""
	return '\n' + m[1].replace('  ', '\t')


def rounds_to_json(rounds: list[Round]) -> str:
	"""Converts a list of rounds to nicely formatted JSON."""
	json_bytes = round_list_adapter.dump_json(rounds, indent=2, exclude_none=True)
	return re.sub(r'\n(\s{2,})', _spaces_to_tabs, json_bytes.decode('utf-8'))
