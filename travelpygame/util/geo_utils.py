from collections.abc import Collection, Iterable, Sequence
from functools import partial
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import numpy
import pyproj
import shapely
from shapely.ops import transform

from .distance import geod_distance, geod_distance_and_bearing, wgs84_geod

if TYPE_CHECKING:
	from shapely.geometry.base import BaseGeometry


def get_poly_vertices(poly: shapely.Polygon | shapely.MultiPolygon) -> list[shapely.Point]:
	if isinstance(poly, shapely.MultiPolygon):
		return list(chain.from_iterable(get_poly_vertices(part) for part in poly.geoms))
	out = shapely.points(poly.exterior.coords)
	if isinstance(out, shapely.Point):
		return [out]
	return out.tolist()


def get_midpoint(point_a: shapely.Point, point_b: shapely.Point):
	# TODO: Alternate calculation that matches what other tools of this nature do
	# TODO: Vectorized version
	forward_azimuth, _, dist = wgs84_geod.inv(
		point_a.x, point_a.y, point_b.x, point_b.y, return_back_azimuth=False
	)
	lng, lat, _ = wgs84_geod.fwd(point_a.x, point_a.y, forward_azimuth, dist / 2)
	return shapely.Point(lng, lat)


def get_antipode(lat: float, lng: float):
	antilat = -lat
	antilng = lng + 180
	if antilng > 180:
		antilng -= 360
	return antilat, antilng


def get_antipodes(lats: 'numpy.ndarray', lngs: 'numpy.ndarray'):
	"""Vectorized version of get_antipode"""
	antilat = -lats
	antilng = lngs + 180
	antilng[antilng > 180] -= 360
	return antilat, antilng


def get_closest_point(
	target_point: shapely.Point, points: Collection[shapely.Point] | shapely.MultiPoint
):
	"""Finds the closest point and the distance to it in a collection of points. Uses geodetic distance. If multiple points are equally close, arbitrarily returns one of them.

	Returns:
		Point, distance in metres
	"""
	if isinstance(points, shapely.MultiPoint):
		points = list(points.geoms)
	generator = ((p, geod_distance(target_point, p)) for p in points)
	return min(generator, key=itemgetter(1))


def get_closest_points(
	target_point: shapely.Point,
	points: 'Sequence[shapely.Point] | shapely.MultiPoint | numpy.ndarray',
):
	"""Finds the closest point(s) and the distance to them in a collection of points. Uses geodetic distance.

	Returns:
		Points, distance in metres
	"""
	# This code kinda sucks I'm sorry
	if isinstance(points, shapely.MultiPoint):
		points = list(points.geoms)
	n = len(points)
	lngs, lats = shapely.get_coordinates(points).T
	target_lng = [target_point.x] * n
	target_lat = [target_point.y] * n
	distances, _ = geod_distance_and_bearing(target_lat, target_lng, lats, lngs)
	shortest = min(distances)
	return [point for i, point in enumerate(points) if distances[i] == shortest], shortest


def get_metric_crs(g: 'BaseGeometry'):
	# It would be more ideal if we could use geopandas estimate_utm_crs, but is it worth creating a temporary GeoSeries for that…
	point = g if isinstance(g, shapely.Point) else g.representative_point()
	return pyproj.CRS(
		f'+proj=aeqd +lat_0={point.y} +lon_0={point.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
	)


def get_centroid(g: 'BaseGeometry', crs: Any = None):
	"""Gets the centroid of some points in WGS84 properly, accounting for projection by converting to a different CRS instead."""
	if not crs:
		crs = get_metric_crs(g)
	transformer = pyproj.Transformer.from_crs('WGS84', crs, always_xy=True)
	projected = transform(transformer.transform, g)
	centroid = projected.centroid
	return transform(partial(transformer.transform, direction='inverse'), centroid)


def circular_mean(angles: Sequence[float] | numpy.ndarray) -> float:
	"""Assumes this is in radians

	Returns:
		Mean angle"""
	if not isinstance(angles, numpy.ndarray):
		angles = numpy.asarray(angles)
	sin_sum = numpy.sin(angles).sum()
	cos_sum = numpy.cos(angles).sum()
	# Convert it from numpy.floating to float otherwise that's maybe annoying
	return float(numpy.atan2(sin_sum, cos_sum))


def circular_mean_xy(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
	"""x and y are assumed to be convertible to numpy.ndarray! I can't be arsed type hinting list-like

	Returns:
		mean of x, mean of y
		i.e. long and then lat, do not get them swapped around I swear on me mum
	"""
	if not isinstance(x, (numpy.ndarray)):
		x = numpy.asarray(x)
	if not isinstance(y, (numpy.ndarray)):
		y = numpy.asarray(y)
	x = numpy.radians(x + 180)
	y = numpy.radians((y + 90) * 2)
	mean_x = numpy.degrees(circular_mean(x))
	mean_y = numpy.degrees(circular_mean(y))
	mean_x = (mean_x % 360) - 180
	mean_y = ((mean_y % 360) / 2) - 90
	return mean_x, mean_y


def circular_mean_points(points: Iterable[shapely.Point]) -> shapely.Point:
	"""points is assumed to be convertible to numpy.ndarray!"""
	x, y = zip(*((a.x, a.y) for a in points), strict=True)
	mean_x, mean_y = circular_mean_xy(x, y)
	return shapely.Point(mean_x, mean_y)
