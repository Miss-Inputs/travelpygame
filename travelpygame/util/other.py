from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pandas import DataFrame
	from shapely import Point


def format_xy(x: float, y: float, decimal_places: int | None = 6) -> str:
	"""Formats x and y coordinates. 6 decimal places should be more than enough for anybody, see also: https://xkcd.com/2170/"""
	# This could potentially have an argument for using that weird northing/easting format instead of decimal degrees
	if decimal_places:
		return f'{format_number(y, decimal_places)}, {format_number(x, decimal_places)}'
	return f'{y}, {x}'


def format_point(p: 'Point', decimal_places: int | None = 6) -> str:
	"""Formats point geometries more nicely than builtin WKT representation. 6 decimal places should be more than enough for anybody, see also: https://xkcd.com/2170/"""
	return format_xy(p.x, p.y, decimal_places)


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
	if abs(n) > 1_000:
		return f'{format_number(n / 1_000, decimal_places)}k{unit}'
	if abs(n) < 1e-2:
		return f'{format_number(n * 100, decimal_places)}c{unit}'
	return f'{format_number(n, decimal_places)}{unit}'


def format_area(n: float, decimal_places: int = 6, unit: str = 'm²'):
	if abs(n) > 1_000_000:
		return f'{format_number(n / 1_000_000, decimal_places)}k{unit}'
	return f'{format_number(n, decimal_places)}{unit}'


def _format_dataframe_inner(
	df: 'DataFrame', cols: Iterable[Hashable] | Hashable | None, formatter: Callable[..., str]
):
	if cols is None:
		return
	if isinstance(cols, Iterable) and not isinstance(cols, (str, bytes)):
		for col in cols:
			df[col] = df[col].map(formatter)
	else:
		df[cols] = df[cols].map(formatter)


def format_dataframe(
	df: 'DataFrame',
	distance_cols: Iterable[Hashable] | Hashable | None = None,
	point_cols: Iterable[Hashable] | Hashable | None = None,
	area_cols: Iterable[Hashable] | Hashable | None = None,
	number_cols: Iterable[Hashable] | Hashable | None = None,
	*,
	copy: bool = True,
) -> 'DataFrame':
	if copy:
		df = df.copy()
	_format_dataframe_inner(df, distance_cols, format_distance)
	_format_dataframe_inner(df, point_cols, format_point)
	_format_dataframe_inner(df, area_cols, format_area)
	_format_dataframe_inner(df, number_cols, format_number)
	return df
