"""Utilities to get extreme points within a polygon."""

from collections.abc import Iterable
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
	if not is_already_projected:
		if not metric_crs:
			metric_crs = get_metric_crs(geom)
		trans_to, trans_from = get_transform_methods(crs, metric_crs)
		geom = shapely.ops.transform(trans_to, geom)
	else:
		trans_from = None
	shapely.prepare(geom)

	minx, miny, maxx, maxy = geom.bounds
	nw = shapely.Point(minx, maxy)
	ne = shapely.Point(maxx, maxy)
	se = shapely.Point(maxx, miny)
	sw = shapely.Point(minx, miny)

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
