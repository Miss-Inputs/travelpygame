"""Utilities to get extreme points within a polygon."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import geopandas
import numpy
import pandas
import shapely
import shapely.ops

from .distance import geod_distance_and_bearing
from .geo_utils import (
	circular_mean_xy,
	fix_x_coord,
	fix_y_coord,
	get_metric_crs,
	get_transform_methods,
)

if TYPE_CHECKING:
	from shapely.geometry.base import BaseGeometry


def _add_points(
	d: dict[str, shapely.Point], name: str, coords: numpy.ndarray | tuple[float, float]
):
	if isinstance(coords, tuple) or coords.ndim == 1:
		d[name] = shapely.Point(coords)
	elif len(coords) == 1:
		d[name] = shapely.Point(coords[0])
	else:
		points = shapely.points(coords)
		assert isinstance(points, numpy.ndarray), f'points is {type(points)}'
		d.update((f'{name} {i}', point) for i, point in enumerate(points, 1))


def _drop_duplicates(gs: geopandas.GeoSeries) -> geopandas.GeoSeries:
	s = gs.drop_duplicates()
	assert isinstance(s, geopandas.GeoSeries), f'_drop_duplicates returned {type(s)}'
	return s


def _concat_no_dupes(serieses: Iterable[geopandas.GeoSeries]) -> geopandas.GeoSeries:
	s = pandas.concat(serieses, axis='index', verify_integrity=True).drop_duplicates()
	assert isinstance(s, geopandas.GeoSeries), f'_concat_no_dupes returned {type(s)}'
	return s


def _maybe_prefix(s1: str | None, s2: Any) -> str:
	return f'{s1} {s2}' if s1 else str(s2)


def get_extreme_points(
	geom: 'BaseGeometry',
	crs: Any | None = 'wgs84',
	name: str | None = None,
	*,
	find_centre_points: bool = False,
	include_multipolygon_parts: bool = False,
	include_interiors: bool = True,
	force_non_contained_centre_points: bool = False,
) -> geopandas.GeoSeries:
	"""Gets extreme points of a geometry (which would generally be a polygon or multipolygon).

	Arguments:
		geom: Geometry, which would generally be a polygon or multipolygon.
		crs: The CRS of the returned GeoSeries, which should be the CRS that geom's coordinates are in.
		name: Name of geom, to append to the names of each point.
		find_centre_points: If true, include points that are kind of the centre, but not really.

	Returns:
		GeoSeries containing points, with index = name of each point ("westmost", "northmost", etc.)
	"""
	if isinstance(geom, shapely.MultiPolygon) and include_multipolygon_parts:
		return _concat_no_dupes(
			get_extreme_points(poly, crs, _maybe_prefix(name, i))
			for i, poly in enumerate(geom.geoms, 1)
		)
	if isinstance(geom, shapely.Polygon):
		exterior = get_extreme_points(geom.exterior, crs, name)
		if not include_interiors:
			return exterior
		if len(geom.interiors) == 1:
			return _concat_no_dupes(
				(
					exterior,
					get_extreme_points(geom.interiors[0], crs, _maybe_prefix(name, 'interior')),
				)
			)
		return _concat_no_dupes(
			(
				exterior,
				*(
					get_extreme_points(interior, crs, _maybe_prefix(name, f'interior {i}'))
					for i, interior in enumerate(geom.interiors, 1)
				),
			)
		)

	coords = numpy.unique(shapely.get_coordinates(geom), axis=0, sorted=False)
	x, y = coords.T
	max_west = x.min()
	max_east = x.max()
	max_south = y.min()
	max_north = y.max()

	d: dict[str, shapely.Point] = {}
	_add_points(d, _maybe_prefix(name, 'westmost point'), coords[x == max_west])
	_add_points(d, _maybe_prefix(name, 'eastmost point'), coords[x == max_east])
	_add_points(d, _maybe_prefix(name, 'northmost point'), coords[y == max_north])
	_add_points(d, _maybe_prefix(name, 'southmost point'), coords[y == max_south])
	if find_centre_points:
		mean_x, mean_y = circular_mean_xy(x, y)
		if force_non_contained_centre_points or shapely.intersects_xy(geom, mean_x, mean_y):
			_add_points(d, _maybe_prefix(name, 'boundary circular mean'), (mean_x, mean_y))

		# This is not _really_ correct btw but eh
		centre_x = fix_x_coord((max_west + max_east) / 2)
		centre_y = fix_y_coord((max_south + max_north) / 2)
		if force_non_contained_centre_points or shapely.intersects_xy(geom, centre_x, centre_y):
			_add_points(d, _maybe_prefix(name, 'centre of extremes'), (centre_x, centre_y))

	gs = geopandas.GeoSeries(d, crs=crs)
	return _drop_duplicates(gs)


def get_extreme_corner_vertices(
	geom: 'BaseGeometry', crs: Any | None = 'wgs84', name: str | None = None
) -> geopandas.GeoSeries:
	"""Tries to find northwest/northeast/southeast/southwest points, by getting the closest vertex to each corner of the bounding box. Might not be entirely correct, as it can only ever return vertices in the original geometry, and might be a bit slow.

	Arguments:
		geom: Geometry, intended to be a polygon/multipolygon.
		crs: The CRS of the returned GeoSeries, which should be the CRS that geom's coordinates are in.
		name: Name of geom, to append to the names of each point.
	"""
	minx, miny, maxx, maxy = geom.bounds

	coords = numpy.unique(shapely.get_coordinates(geom), axis=0, sorted=False)
	# Theoretically we should be able to speed this up by excluding coordinates which are not the right answer, who knows
	x, y = coords.T
	n = len(coords)

	min_x = numpy.repeat(minx, n)
	max_x = numpy.repeat(maxx, n)
	min_y = numpy.repeat(miny, n)
	max_y = numpy.repeat(maxy, n)

	nw_distances = geod_distance_and_bearing(max_y, min_x, y, x)[0]
	ne_distances = geod_distance_and_bearing(max_y, max_x, y, x)[0]
	se_distances = geod_distance_and_bearing(min_y, max_x, y, x)[0]
	sw_distances = geod_distance_and_bearing(min_y, min_x, y, x)[0]

	nw_most = coords[nw_distances.argmin()]
	ne_most = coords[ne_distances.argmin()]
	se_most = coords[se_distances.argmin()]
	sw_most = coords[sw_distances.argmin()]

	d = {
		_maybe_prefix(name, 'northwestmost point'): shapely.Point(nw_most),
		_maybe_prefix(name, 'northeastmost point'): shapely.Point(ne_most),
		_maybe_prefix(name, 'southeastmost point'): shapely.Point(se_most),
		_maybe_prefix(name, 'southwestmost point'): shapely.Point(sw_most),
	}
	gs = geopandas.GeoSeries(d, crs=crs)
	return _drop_duplicates(gs)


def get_extreme_corner_points(
	geom: 'BaseGeometry',
	crs: Any | None = 'wgs84',
	name: str | None = None,
	metric_crs: Any | None = None,
	*,
	is_already_projected: bool = False,
) -> geopandas.GeoSeries:
	"""Tries to find northwest/northeast/southeast/southwest points, by getting the closest point to each corner of the bounding box. Might not be entirely correct, and might be a bit slow. This is a different method

	Arguments:
		geom: Geometry, intended to be a polygon/multipolygon.
		crs: The CRS of the returned GeoSeries, which should be the CRS that geom's coordinates are in.
		name: Name of geom, to append to the names of each point.
	"""
	minx, miny, maxx, maxy = geom.bounds
	if not is_already_projected:
		if not metric_crs:
			metric_crs = get_metric_crs(geom)
		trans_to, trans_from = get_transform_methods(crs, metric_crs)
		geom = shapely.ops.transform(trans_to, geom)
	else:
		trans_from = None
		trans_to = None
	shapely.prepare(geom)

	nw = shapely.Point(minx, maxy)
	ne = shapely.Point(maxx, maxy)
	se = shapely.Point(maxx, miny)
	sw = shapely.Point(minx, miny)
	if trans_to:
		nw = shapely.ops.transform(trans_to, nw)
		ne = shapely.ops.transform(trans_to, ne)
		se = shapely.ops.transform(trans_to, se)
		sw = shapely.ops.transform(trans_to, sw)

	nw_most = shapely.ops.nearest_points(geom, nw)[0]
	ne_most = shapely.ops.nearest_points(geom, ne)[0]
	se_most = shapely.ops.nearest_points(geom, se)[0]
	sw_most = shapely.ops.nearest_points(geom, sw)[0]
	if trans_from:
		nw_most = shapely.ops.transform(trans_from, nw_most)
		ne_most = shapely.ops.transform(trans_from, ne_most)
		se_most = shapely.ops.transform(trans_from, se_most)
		sw_most = shapely.ops.transform(trans_from, sw_most)

	d = {
		_maybe_prefix(name, 'northwestmost point'): nw_most,
		_maybe_prefix(name, 'northeastmost point'): ne_most,
		_maybe_prefix(name, 'southeastmost point'): se_most,
		_maybe_prefix(name, 'southwestmost point'): sw_most,
	}
	gs = geopandas.GeoSeries(d, crs=crs)
	return _drop_duplicates(gs)


def get_extreme_corners_of_point_set(gs: geopandas.GeoSeries):
	"""Special case where get_extreme_corner_vertices would otherwise be used, but just gets the indexes of the points in `gs` that are closest to corners of the bounding box.

	Returns:
		northwestmost index, northeastmost index, southeastmost index, southwestmost index
	"""
	minx, miny, maxx, maxy = gs.total_bounds

	coords = shapely.get_coordinates(gs)
	x, y = coords.T
	n = len(coords)

	min_x = numpy.repeat(minx, n)
	max_x = numpy.repeat(maxx, n)
	min_y = numpy.repeat(miny, n)
	max_y = numpy.repeat(maxy, n)

	nw_distances = geod_distance_and_bearing(max_y, min_x, y, x)[0]
	ne_distances = geod_distance_and_bearing(max_y, max_x, y, x)[0]
	se_distances = geod_distance_and_bearing(min_y, max_x, y, x)[0]
	sw_distances = geod_distance_and_bearing(min_y, min_x, y, x)[0]

	nw_most = nw_distances.argmin().item()
	ne_most = ne_distances.argmin().item()
	se_most = se_distances.argmin().item()
	sw_most = sw_distances.argmin().item()

	return gs.index[nw_most], gs.index[ne_most], gs.index[se_most], gs.index[sw_most]


def _get_row_col(i: int, x_size: int, y_size: int, *, reverse_y: bool = True) -> tuple[int, int]:
	# Probably a better way to do that than divmod but eh, if it works it works
	y, x = divmod(i, x_size)
	return x, (y_size - y) - 1 if reverse_y else y


def get_grid(
	x: numpy.ndarray | Sequence[float],
	y: numpy.ndarray | Sequence[float],
	crs: Any = 'wgs84',
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoSeries:
	"""Creates a grid of points from x and y coordinates. Index of the returned GeoSeries will be a MultiIndex with level 0 = counting upwards for every lng/x and level 1 = counting upwards for every lat/y.

	Arguments:
		x: Sequence/1D array of x/lng coordinates to form the grid.
		y: Sequence/1D array of y/lat coordinates to form the grid.
		crs: CRS of returned GeoSeries.
		reverse_y_in_index: Start counting from the maximum y instead of the mininum (this is likely more intuitive for grids in WGS84 coordinates as minimum y will be south and not north.)
	"""
	x_size = numpy.size(x)
	y_size = numpy.size(y)

	x_grid, y_grid = numpy.meshgrid(x, y)
	grid = numpy.column_stack((x_grid.ravel(), y_grid.ravel()))
	"""2D array of shape (xsize * y_size, 2)"""
	points = shapely.points(grid)
	assert not isinstance(points, shapely.Point), 'why is points a single point'

	rowcols = pandas.MultiIndex.from_tuples(
		[
			(_get_row_col(i, x_size, y_size, reverse_y=reverse_y_in_index))
			for i in range(x_grid.size)
		],
		names=('x', 'y'),
	)
	return geopandas.GeoSeries(points, rowcols, crs)


def get_fixed_grid(
	min_x: float,
	min_y: float,
	max_x: float,
	max_y: float,
	resolution: float | tuple[float, float],
	crs: Any = 'wgs84',
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoSeries:
	"""Creates a grid of points of a fixed distance. Index of the returned GeoSeries will be a MultiIndex with level 0 = counting upwards for every lng/x and level 1 = counting upwards for every lat/y.

	Arguments:
		min_x, min_y, max_x, max_y: Boundary of grid.
		resolution: Distance between each point on each axis in the units of the coordinate system (e.g. for WGS84, 1.0 will produce a grid of points every 1 latitude and 1 longitude.)
		crs: CRS of returned GeoSeries.
		reverse_y_in_index: Start counting from the maximum y instead of the mininum (this is likely more intuitive for grids in WGS84 coordinates as minimum y will be south and not north.)
	"""
	if isinstance(resolution, tuple):
		x_res, y_res = resolution
	else:
		x_res = y_res = resolution
	x = numpy.arange(min_x, max_x, x_res)
	y = numpy.arange(min_y, max_y, y_res)
	return get_grid(x, y, crs, reverse_y_in_index=reverse_y_in_index)


def get_spaced_grid(
	min_x: float,
	min_y: float,
	max_x: float,
	max_y: float,
	amount: int | tuple[int, int],
	crs: Any = 'wgs84',
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoSeries:
	"""Creates a grid of a specific amount of points, evenly spaced. Index of the returned GeoSeries will be a MultiIndex with level 0 = counting upwards for every lng/x and level 1 = counting upwards for every lat/y.

	Arguments:
		min_x, min_y, max_x, max_y: Boundary of grid.
		amount: Amount of points in each direction (i.e. output will contain amount ** 2 points.)
		crs: CRS of returned GeoSeries.
		reverse_y_in_index: Start counting from the maximum y instead of the mininum (this is likely more intuitive for grids in WGS84 coordinates as minimum y will be south and not north.)
	"""
	if isinstance(amount, tuple):
		x_amount, y_amount = amount
	else:
		x_amount = y_amount = amount
	x = numpy.linspace(min_x, max_x, x_amount)
	y = numpy.linspace(min_y, max_y, y_amount)
	return get_grid(x, y, crs, reverse_y_in_index=reverse_y_in_index)


def get_grid_over_geodataframe(
	gdf: geopandas.GeoDataFrame,
	x: numpy.ndarray | Sequence[float],
	y: numpy.ndarray | Sequence[float],
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoDataFrame:
	grid = get_grid(x, y, gdf.crs, reverse_y_in_index=reverse_y_in_index)
	indices = gdf.sindex.query(grid, 'intersects', output_format='indices')

	# Alright this might be little a convoluted
	points = []
	x_indices = []
	y_indices = []
	gdf_indices = []
	for point_index, poly_index in indices.T:
		x, y = grid.index[point_index]
		x_indices.append(x)
		y_indices.append(y)
		points.append(grid.iloc[point_index])
		gdf_indices.append(gdf.index[poly_index])
	return geopandas.GeoDataFrame(
		{'x': x_indices, 'y': y_indices, gdf.index.name or 'index': gdf_indices},
		geometry=points,
		crs=gdf.crs,
	)


def get_fixed_grid_over_geodataframe(
	gdf: geopandas.GeoDataFrame,
	bounds: tuple[float, float, float, float] | None,
	resolution: float | tuple[float, float],
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoDataFrame:
	min_x, min_y, max_x, max_y = gdf.total_bounds if bounds is None else bounds
	if isinstance(resolution, tuple):
		x_res, y_res = resolution
	else:
		x_res = y_res = resolution
	x = numpy.arange(min_x, max_x, x_res)
	y = numpy.arange(min_y, max_y, y_res)
	return get_grid_over_geodataframe(gdf, x, y, reverse_y_in_index=reverse_y_in_index)


def get_spaced_grid_over_geodataframe(
	gdf: geopandas.GeoDataFrame,
	bounds: tuple[float, float, float, float] | None,
	amount: int | tuple[int, int],
	*,
	reverse_y_in_index: bool = True,
) -> geopandas.GeoDataFrame:
	min_x, min_y, max_x, max_y = gdf.total_bounds if bounds is None else bounds
	if isinstance(amount, tuple):
		x_amount, y_amount = amount
	else:
		x_amount = y_amount = amount
	x = numpy.linspace(min_x, max_x, x_amount)
	y = numpy.linspace(min_y, max_y, y_amount)
	return get_grid_over_geodataframe(gdf, x, y, reverse_y_in_index=reverse_y_in_index)


_Corner = list[float]


def get_box_grid(
	x: numpy.ndarray | Sequence[float],
	y: numpy.ndarray | Sequence[float],
	crs: Any = 'wgs84',
	*,
	reverse_y: bool = True,
) -> geopandas.GeoSeries:
	"""Creates a grid of rectangles from x and y coordinates. Index of the returned GeoSeries will be a MultiIndex with level 0 = counting upwards for every lng/x and level 1 = counting upwards for every lat/y.

	Arguments:
		x: Sequence/1D array of x/lng coordinates to form the grid.
		y: Sequence/1D array of y/lat coordinates to form the grid.
		crs: CRS of returned GeoSeries.
		reverse_y: Start counting from the maximum y instead of the mininum (this is likely more intuitive for grids in WGS84 coordinates as minimum y will be south and not north.)
	"""
	x_size = numpy.size(x)
	y_size = numpy.size(y)

	x_grid, y_grid = numpy.meshgrid(x, y)
	grid = numpy.column_stack((x_grid.ravel(), y_grid.ravel()))
	"""2D array of shape (xsize * y_size, 2)"""
	polygon_coords: list[tuple[_Corner, _Corner, _Corner, _Corner, _Corner]] = []
	# There is a better way to do this with fancy slicing probably, but I've overthunked this for too long and don't know _that_ much what I'm doing
	for i in range(len(grid) - (x_size + 1)):
		if i % x_size == (x_size - 1):
			continue
		sw_corner = grid[i]
		se_corner = grid[i + 1]
		nw_corner = grid[i + x_size]
		ne_corner = grid[i + x_size + 1]
		box = (sw_corner, nw_corner, ne_corner, se_corner, sw_corner)
		polygon_coords.append(box)
	boxes = shapely.polygons(polygon_coords)

	rowcols = pandas.MultiIndex.from_tuples(
		[(_get_row_col(i, x_size, y_size, reverse_y=reverse_y)) for i in range(len(boxes))],
		names=('x', 'y'),
	)
	return geopandas.GeoSeries(boxes, rowcols, crs)


def get_fixed_box_grid(
	min_x: float,
	min_y: float,
	max_x: float,
	max_y: float,
	resolution: float | tuple[float, float],
	crs: Any = 'wgs84',
	*,
	reverse_y: bool = True,
) -> geopandas.GeoSeries:
	"""Creates a grid of boxes of a fixed distance. Index of the returned GeoSeries will be a MultiIndex with level 0 = counting upwards for every lng/x and level 1 = counting upwards for every lat/y.

	Arguments:
		min_x, min_y, max_x, max_y: Boundary of grid.
		resolution: Distance between each point on each axis in the units of the coordinate system (e.g. for WGS84, 1.0 will produce a grid of points every 1 latitude and 1 longitude.)
		crs: CRS of returned GeoSeries.
		reverse_y: Start counting from the maximum y instead of the mininum (this is likely more intuitive for grids in WGS84 coordinates as minimum y will be south and not north.)
	"""
	if isinstance(resolution, tuple):
		x_res, y_res = resolution
	else:
		x_res = y_res = resolution
	x = numpy.arange(min_x, max_x, x_res)
	y = numpy.arange(min_y, max_y, y_res)
	return get_box_grid(x, y, crs, reverse_y=reverse_y)
