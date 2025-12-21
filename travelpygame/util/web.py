from aiohttp import ClientSession, ClientTimeout

user_agent = 'https://github.com/Miss-Inputs/travelpygame'


async def get_text(url: str, session: ClientSession|None=None, client_timeout:ClientTimeout|float|None=60.0)->str:
	if session is None:
		async with ClientSession(headers={'User-Agent': user_agent}) as sesh:
			return await get_text(url, sesh, client_timeout)
	timeout=ClientTimeout(client_timeout)if isinstance(client_timeout,(float, int)) else client_timeout
	async with session.get(url, timeout=timeout) as response:
		response.raise_for_status()
		return await response.text()