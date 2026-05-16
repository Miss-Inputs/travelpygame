"""Utilities to work with geometries."""

from collections.abc import Hashable, Iterable, Sequence
from itertools import chain

import numpy
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry


def get_poly_vertices(poly: shapely.Polygon | shapely.MultiPolygon) -> list[shapely.Point]:
	if isinstance(poly, shapely.MultiPolygon):
		return list(chain.from_iterable(get_poly_vertices(part) for part in poly.geoms))
	out = shapely.points(poly.exterior.coords)
	if isinstance(out, shapely.Point):
		return [out]
	return out.tolist()


def contains_any(
	geo: GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry,
	point: shapely.Point | tuple[float, float],
) -> bool:
	"""Returns a bool indicating if a point is anywhere within any part of a geometry or GeoPandas object."""
	if isinstance(geo, BaseGeometry):
		return (
			shapely.contains_xy(geo, point[1], point[0]).item()
			if isinstance(point, tuple)
			else geo.contains(point)
		)
	if isinstance(point, tuple):
		point = shapely.Point(point)
	# Unfortunately there is no contains_properly inverse
	return geo.sindex.query(point, 'within').size > 0


def contains_any_array(
	geo: GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry,
	points: numpy.ndarray | Sequence[shapely.Point],
):
	"""Returns an array of of booleans for each point, indicating whether each point is anywhere in a geometry or GeoPandas object."""
	if isinstance(geo, BaseGeometry):
		return geo.contains(points)
	# Unfortunately there is no contains_properly inverse
	if not isinstance(points, numpy.ndarray):
		points = numpy.asarray(points)
	return geo.sindex.query(points, 'within', output_format='dense').any(axis=0)


def get_polygons(
	geom: GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry,
) -> list[shapely.Polygon]:
	"""Gets every individual polygon inside geom."""
	if isinstance(geom, shapely.Polygon):
		return [geom]
	if isinstance(geom, BaseMultipartGeometry):
		return list(chain.from_iterable(get_polygons(part) for part in geom.geoms))
	if isinstance(geom, GeoDataFrame):
		geom = geom.geometry
	if not isinstance(geom, BaseGeometry):
		# GeoSeries/array/etc
		return list(chain.from_iterable(get_polygons(item) for item in geom.dropna()))
	# Some other geometry, just silently return nothing
	return []


def find_first_geom_index(
	gdf: GeoDataFrame | GeoSeries, geom: 'BaseGeometry', tolerance: float | None = 1e-6
) -> Hashable | None:
	if isinstance(gdf, GeoDataFrame):
		gdf = gdf.geometry
	if geom in gdf or tolerance is None:
		rows = gdf[gdf == geom]
	else:
		rows = gdf[gdf.geom_equals_exact(geom, tolerance)]
	if rows.empty:
		return None
	return rows.index[0]


def get_total_bounds(
	geoms: Iterable[BaseGeometry] | GeoSeries | GeoDataFrame | GeometryArray,
) -> tuple[float, float, float, float]:
	"""Shortcut for Geo*.total_bounds, but also works with any iterable of geometries.

	Returns:
		min x (lng), min y (lat), max x (lng), max y (lat)
	"""
	if isinstance(geoms, (GeoSeries, GeoDataFrame, GeometryArray)):
		total_bounds = geoms.total_bounds
		# Ensures it ends up being float instead of numpy.floating, because I'm just petty like that
		min_x, min_y, max_x, max_y = total_bounds.tolist()
	else:
		all_min_x, all_min_y, all_max_x, all_max_y = zip(
			*(geom.bounds for geom in geoms), strict=True
		)
		min_x = min(all_min_x)
		min_y = min(all_min_y)
		max_x = max(all_max_x)
		max_y = max(all_max_y)
	return min_x, min_y, max_x, max_y


def get_bbox_corners(
	geoms: Iterable[BaseGeometry] | GeoSeries | GeoDataFrame | GeometryArray,
) -> tuple[shapely.Point, shapely.Point, shapely.Point, shapely.Point]:
	"""Returns corner points of the bounding box of `geoms` as a tuple:
	min_x/min_y (southwest)
	max_x/min_y (southeast)
	min_x/max_y (northwest)
	max_x/max_y (northeast)
	"""
	min_x, min_y, max_x, max_y = get_total_bounds(geoms)
	sw = shapely.Point(min_x, min_y)
	se = shapely.Point(max_x, min_y)
	nw = shapely.Point(min_x, max_y)
	ne = shapely.Point(max_x, max_y)
	return sw, se, nw, ne
