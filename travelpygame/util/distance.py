"""Tools for measuring distance and such."""

from collections.abc import Collection, Sequence
from operator import itemgetter
from typing import overload

import numpy
import pandas
import pyproj
import shapely

wgs84_geod = pyproj.Geod(ellps='WGS84')

FloatListlike = Sequence[float] | numpy.ndarray | pandas.Series
"""Accepted input types to pyproj.Geod.inv, although other stuff would probably work, this is just what works as a type hint."""


@overload
def geod_distance_and_bearing(
	lat1: float, lng1: float, lat2: float, lng2: float, *, radians: bool = False
) -> tuple[float, float]: ...


@overload
def geod_distance_and_bearing(
	lat1: FloatListlike,
	lng1: FloatListlike,
	lat2: FloatListlike,
	lng2: FloatListlike,
	*,
	radians: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]: ...


def geod_distance_and_bearing(
	lat1: float | FloatListlike,
	lng1: float | FloatListlike,
	lat2: float | FloatListlike,
	lng2: float | FloatListlike,
	*,
	radians: bool = False,
) -> tuple[float | numpy.ndarray, float | numpy.ndarray]:
	"""
	Calculates the WGS84 geodesic distance and heading from one point to another. lat1/lng1/lat2/lng2 can either all be floats, or all arrays.

	Arguments:
		lat1: Latitude of point A, or list/ndarray/etc
		lng1: Longitude of point A, or list/ndarray/etc
		lat2: Latitude of point B, or list/ndarray/etc
		lng2: Longitude of point B, or list/ndarray/etc
		radians: If true, treats the arguments as being in radians, otherwise they are degrees (as normal people use for coordinates)

	Returns:
		(Distance in metres, heading/direction/bearing/whatever you call it from lat1,lng1 to lat2,lng2 in degrees/radians) between point A and point B. If input is an array, it will return an array for each pair of coordinates.
	"""
	bearing, _, dist = wgs84_geod.inv(lng1, lat1, lng2, lat2, radians=radians)
	if isinstance(bearing, list):
		# y u do this
		bearing = numpy.array(bearing)
	return (dist, bearing)


def geod_distance(point1: shapely.Point, point2: shapely.Point) -> float:
	"""Returns WGS84 geodesic distance between point1 and point2 (assumed to be WGS84 coordinates) in metres."""
	return geod_distance_and_bearing(point1.y, point1.x, point2.y, point2.x)[0]


@overload
def haversine_distance(
	lat1: float, lng1: float, lat2: float, lng2: float, *, radians: bool = False
) -> float: ...


@overload
def haversine_distance(
	lat1: numpy.ndarray,
	lng1: numpy.ndarray,
	lat2: numpy.ndarray,
	lng2: numpy.ndarray,
	*,
	radians: bool = False,
) -> numpy.ndarray: ...


def haversine_distance(
	lat1: float | numpy.ndarray,
	lng1: float | numpy.ndarray,
	lat2: float | numpy.ndarray,
	lng2: float | numpy.ndarray,
	*,
	radians: bool = False,
) -> float | numpy.ndarray:
	"""Calculates haversine distance (which TPG uses), treating the earth as a sphere.

	Arguments:
		lat1: ndarray of floats
		lng1: ndarray of floats
		lat1: ndarray of floats
		lng1: ndarray of floats
		radians: If set to true, treats the lat/long arguments as being in radians, otherwise they are treated as degrees (as normal people would use for coordinates)

	Returns:
		ndarray (float) of distances in metres

	"""
	r = 6371_000
	if not radians:
		lat1 = numpy.radians(lat1)
		lat2 = numpy.radians(lat2)
		lng1 = numpy.radians(lng1)
		lng2 = numpy.radians(lng2)
	dlng = lng2 - lng1
	dlat = lat2 - lat1
	a = (numpy.sin(dlat / 2) ** 2) + numpy.cos(lat1) * numpy.cos(lat2) * (numpy.sin(dlng / 2) ** 2)
	c = 2 * numpy.asin(numpy.sqrt(a))
	if isinstance(c, numpy.floating):
		# Just to make sure nothing annoying happens elsewhere
		c = c.item()
	return c * r


def geod_distances(
	lat: numpy.ndarray, lng: numpy.ndarray, target_lat: numpy.ndarray, target_lng: numpy.ndarray
) -> numpy.ndarray:
	"""Vectorized get_geod_distance_and_bearing that just gets the distance and not bearing (for symmetry with haversine_distance)."""
	return geod_distance_and_bearing(lat, lng, target_lat, target_lng)[0]


def get_distances(
	target_point: shapely.Point | tuple[float, float],
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray,
	*,
	use_haversine: bool = False,
):
	"""Finds the distances from all points in `points` to `target_point`, in the original order of points. By default, uses geodetic distance. If `target_point` is a tuple, it should be lat, lng."""
	if isinstance(points, numpy.ndarray) and points.dtype.kind == 'f':
		if points.shape[0] == 2:
			lngs, lats = points
		elif points.shape[1] == 2:
			lngs, lats = points.T
		else:
			raise ValueError(
				'If points is a numpy array of floats, it must be 2D, wih one axis having size 2'
			)
	else:
		if isinstance(points, Collection) and not isinstance(points, Sequence):
			points = list(points)
		lngs, lats = shapely.get_coordinates(points).T
	dist_func = geod_distances if use_haversine else haversine_distance
	if isinstance(target_point, shapely.Point):
		target_lat = target_point.y
		target_lng = target_point.x
	else:
		target_lat, target_lng = target_point
	return dist_func(
		numpy.repeat(target_lat, lats.size), numpy.repeat(target_lng, lngs.size), lats, lngs
	)


def get_closest_point(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint,
	*,
	use_haversine: bool = False,
) -> tuple[shapely.Point, float]:
	"""Finds the closest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns one of them.

	Returns:
		Point, distance in metres
	"""
	if isinstance(points, shapely.MultiPoint):
		points = list(points.geoms)
	if isinstance(points, Sequence):
		distances = get_distances(target_point, points, use_haversine=use_haversine)
		index = distances.argmin().item()
		return points[index], distances[index]

	generator = (
		(
			p,
			haversine_distance(target_point.y, target_point.x, p.y, p.x)
			if use_haversine
			else geod_distance(target_point, p),
		)
		for p in points
	)
	return min(generator, key=itemgetter(1))


def get_closest_index(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray,
	*,
	use_haversine: bool = False,
) -> tuple[int, float]:
	"""Finds the index of the closest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns the index of one of them.

	Returns:
		Point, distance in metres
	"""
	distances = get_distances(target_point, points, use_haversine=use_haversine)
	index = distances.argmin().item()
	return index, distances[index]


def get_furthest_index(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray,
	*,
	use_haversine: bool = False,
) -> tuple[int, float]:
	"""Finds the index of the furthest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns the index of one of them.

	Returns:
		Point, distance in metres
	"""
	distances = get_distances(target_point, points, use_haversine=use_haversine)
	index = distances.argmax().item()
	return index, distances[index]


def get_closest_points(
	target_point: shapely.Point,
	points: 'Sequence[shapely.Point] | shapely.MultiPoint | numpy.ndarray',
	*,
	use_haversine: bool = False,
):
	"""Finds the closest point(s) and the distance to them in a collection of points. Uses geodetic distance by default.

	Returns:
		Points, distance in metres
	"""
	if isinstance(points, shapely.MultiPoint):
		points = list(points.geoms)
	n = len(points)
	lngs, lats = shapely.get_coordinates(points).T
	target_lng = numpy.repeat(target_point.x, n)
	target_lat = numpy.repeat(target_point.y, n)
	dist_func = haversine_distance if use_haversine else geod_distances
	distances = dist_func(target_lat, target_lng, lats, lngs)
	shortest = min(distances)
	return [point for i, point in enumerate(points) if distances[i] == shortest], shortest
