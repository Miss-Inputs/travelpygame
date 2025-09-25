from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from shapely import Point


def format_xy(x: float, y: float) -> str:
	# This could potentially have an argument for using that weird northing/easting format instead of decimal degrees
	return f'{y}, {x}'


def format_point(p: 'Point') -> str:
	return format_xy(p.x, p.y)


def get_ordinal(n: int) -> str:
	if 10 <= n % 100 <= 20:
		return 'th'
	return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def format_ordinal(n: float) -> str:
	if not n.is_integer():
		# meh
		return f'{n:.2f}th'
	n = int(n)
	return f'{n}{get_ordinal(n)}'


def format_number(n: float, decimal_places: int = 6):
	"""Stops printing annoying stupid scientific notation which looks ugly and sucks grawwrrrr"""
	if n.is_integer():
		return f'{n:n}'
	return f'{n:,.{decimal_places}f}'.rstrip('0')


def format_distance(n: float, decimal_places: int = 6, unit: str = 'm'):
	if n > 1_000:
		return f'{format_number(n / 1_000, decimal_places)}k{unit}'
	if n < 1e-2:
		return f'{format_number(n * 100, decimal_places)}c{unit}'
	return f'{format_number(n, decimal_places)}{unit}'


def format_area(n: float, decimal_places: int = 6, unit: str = 'm²'):
	if n > 1_000_000:
		return f'{format_number(n / 1_000_000, decimal_places)}k{unit}'
	return f'{format_number(n, decimal_places)}{unit}'
