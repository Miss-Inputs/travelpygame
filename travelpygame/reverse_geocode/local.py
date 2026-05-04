"""Reverse geocode a point using local info (such as polygons representing subdivisions), which may not be detailed, but might be good enough."""

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import pandas
from geopandas import GeoDataFrame, GeoSeries

if TYPE_CHECKING:
	from shapely import Point


def reverse_geocode_regions(
	point: 'Point',
	regions: 'GeoDataFrame',
	col_names: Sequence[Hashable] | None,
	*,
	allow_multiple: bool = False,
) -> dict[Hashable, Any]:
	"""Returns the values of which regions in a GeoDataFrame contain a certain point, as a dict.

	If `col_names` is empty or None, returns a dict with a single item with key = the index name of `regions` (or "index" if that is blank) and value = list of the indexes within regions.

	Arguments:
		allow_multiple: Return a list of values in regions for each column, if regions overlap."""
	indices = regions.sindex.query(point, 'within', output_format='indices')
	if not indices.size:
		return {}
	rows = regions.iloc[indices]
	if not col_names:
		# Maybe it's better to just get every column here, oh well
		indexes = rows.index.to_list()
		return {regions.index.name or 'index': indexes if allow_multiple else indexes[0]}
	rows = rows[col_names]
	return rows.to_dict('list') if allow_multiple else rows.iloc[0].to_dict()


def reverse_geocode_regions_multiple(
	points: 'GeoDataFrame | GeoSeries',
	regions: 'GeoDataFrame',
	col_names: list[Hashable] | None,
	*,
	allow_multiple: bool = False,
) -> pandas.DataFrame:
	"""Returns the values for regions which contain specified points, as a GeoDataFrame with the same index as `points`.

	If `col_names` is empty or None, returns all columns in `regions`.

	Arguments:
		allow_multiple: Return a list of the column value for every region that contains each point, if regions overlap.
	"""
	if isinstance(points, GeoDataFrame):
		points = points.geometry

	# Just indices will do here. Maybe it's not actually simpler…
	indices = regions.sindex.query(points, 'within', output_format='indices')
	data = {}
	for point_i, region_i in zip(*indices, strict=True):
		assert isinstance(point_i, int), f'point_i was {type(point_i)} instead of int, uh oh'
		point_index = points.index[point_i]
		region_rows: pandas.DataFrame = regions.iloc[region_i]
		if col_names:
			region_rows = region_rows[col_names]
		data[point_index] = (
			region_rows.to_dict('list') if allow_multiple else region_rows.iloc[0].to_dict()
		)
	return pandas.DataFrame.from_dict(data, 'index').align(points, join='right', axis='index')[0]
