from typing import Any

from aiohttp import ClientSession, ClientTimeout
from tqdm.auto import tqdm

user_agent = 'https://github.com/Miss-Inputs/travelpygame'


async def get_text(
	url: str,
	session: ClientSession | None = None,
	client_timeout: ClientTimeout | float | None = 60.0,
) -> str:
	if session is None:
		async with ClientSession(headers={'User-Agent': user_agent}) as sesh:
			return await get_text(url, sesh, client_timeout)
	timeout = (
		ClientTimeout(client_timeout)
		if isinstance(client_timeout, (float, int))
		else client_timeout
	)
	async with session.get(url, timeout=timeout) as response:
		response.raise_for_status()
		return await response.text()


async def get_bytes_tqdm(
	url: str,
	session: ClientSession | None = None,
	client_timeout: ClientTimeout | float | None = 60.0,
) -> bytes:
	if session is None:
		async with ClientSession(headers={'User-Agent': user_agent}) as sesh:
			return await get_bytes_tqdm(url, sesh, client_timeout)
	timeout = (
		ClientTimeout(client_timeout)
		if isinstance(client_timeout, (float, int))
		else client_timeout
	)

	chunks = []
	async with session.get(url, timeout=timeout, raise_for_status=True) as response:
		with tqdm(
			desc=f'Downloading {url}', total=response.content_length, unit='iB', unit_scale=True
		) as t:
			async for chunk, _end_of_http in response.content.iter_chunks():
				chunks.append(chunk)
				t.update(len(chunk))
	return b''.join(chunks)


async def get_bytes_streamed(
	url: str,
	params: dict[str, Any],
	session: ClientSession | None = None,
	client_timeout: ClientTimeout | float | None = 60.0,
) -> bytes:
	"""Gets bytes streamt by newline."""
	if session is None:
		async with ClientSession(headers={'User-Agent': user_agent}) as sesh:
			return await get_bytes_streamed(url, params, sesh, client_timeout)
	timeout = (
		ClientTimeout(client_timeout)
		if isinstance(client_timeout, (float, int))
		else client_timeout
	)

	lines = []
	async with session.get(url, params=params, timeout=timeout, raise_for_status=True) as response:
		with tqdm(
			desc=f'Downloading {url}', total=response.content_length, unit='iB', unit_scale=True
		) as t:
			while True:
				line = await response.content.readline()
				if not line:
					break
				lines.append(line)
				t.update(len(line))
	return b''.join(lines)  # readline already includes the separator
