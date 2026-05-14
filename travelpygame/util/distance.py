"""Tools for measuring distance and such."""

from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from itertools import combinations
from operator import itemgetter
from typing import overload

import numpy
import pandas
import pyproj
import shapely
from geopandas import GeoSeries

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


def geod_distance(
	point1: shapely.Point | tuple[float, float], point2: shapely.Point | tuple[float, float]
) -> float:
	"""Returns WGS84 geodesic distance between point1 and point2 (assumed to be WGS84 coordinates) in metres. If any arguments are specified as tuples, they are (lat, lng), not the other way around."""
	if isinstance(point1, shapely.Point):
		lat1 = point1.y
		lng1 = point1.x
	else:
		lat1, lng1 = point1
	if isinstance(point2, shapely.Point):
		lat2 = point2.y
		lng2 = point2.x
	else:
		lat2, lng2 = point2
	return geod_distance_and_bearing(lat1, lng1, lat2, lng2)[0]


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
		lat2: ndarray of floats
		lng2: ndarray of floats
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
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray | GeoSeries,
	*,
	use_haversine: bool = False,
):
	"""Finds the distances from all points in `points` to `target_point`, in the original order of points. By default, uses geodetic distance. If `target_point` is a tuple, it should be lat, lng.

	Returns:
		1D numpy array of shape (len(points), ) containing distances in metres."""
	if isinstance(points, numpy.ndarray) and points.dtype.kind == 'f':
		if points.shape[0] == 2:
			lngs, lats = points  # ty: ignore[not-iterable] #yes it is, it's just typed weirdly
		elif points.shape[1] == 2:
			lngs, lats = points.T  # ty: ignore[not-iterable] #yes it is, it's just typed weirdly
		else:
			raise ValueError(
				'If points is a numpy array of floats, it must be 2D, wih one axis having size 2'
			)
	else:
		if isinstance(points, Collection) and not isinstance(points, (Sequence, GeoSeries)):
			points = list(points)  # ty:ignore[invalid-assignment] #it is narrowing the return type of list(points) to list[object], which I guess technically could happen if it was passed in as a numpy array of not-floats
		lngs, lats = shapely.get_coordinates(points).T  # ty:ignore[invalid-argument-type] #points should have been narrowed to list[Point] instead of Collection[Point]
	dist_func = haversine_distance if use_haversine else geod_distances
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
) -> tuple[list[shapely.Point], float]:
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
	shortest = distances.min().item()
	return [point for i, point in enumerate(points) if distances[i] == shortest], shortest


def self_cartesian_product_distances(gs: GeoSeries, *, use_haversine: bool = False):
	"""Distances from every point in `gs` to every other point. Tries to be as efficient as possible. Probably isn't.

	Returns:
		dict of dicts, with keys = `gs` index."""
	coords = shapely.get_coordinates(gs)
	distances: defaultdict[Hashable, dict[Hashable, float]] = defaultdict(dict)

	from_indexes, to_indexes = zip(*combinations(range(gs.index.size), 2), strict=True)
	lats = coords[from_indexes, 1]
	lngs = coords[from_indexes, 0]
	lats2 = coords[to_indexes, 1]
	lngs2 = coords[to_indexes, 0]

	dist_func = haversine_distance if use_haversine else geod_distances
	half_distances = dist_func(lats, lngs, lats2, lngs2)
	for i, distance in enumerate(half_distances):
		from_i = gs.index[from_indexes[i]]
		to_i = gs.index[to_indexes[i]]
		distances[from_i][to_i] = distance
		distances[to_i][from_i] = distance
	return distances


def cartesian_product_distances(
	gs_from: GeoSeries, gs_to: GeoSeries, *, use_haversine: bool = False
):
	"""Distances from every point in `gs_from` to every point in `gs_to`. Tries to be as efficient as possible. Probably isn't.

	Results are undefined if either object has an index that is multi-level or not unique, it will probably just not work.

	Arguments:
		gs_from: GeoSeries, geometries must be points.
		gs_to: GeoSeries, geometries must be points.
		use_haversine: Use haversine instead of geodetic distance, defaults to False.

	Returns:
		DataFrame with the index of `gs_from`, each row containing distances (in metres) to each point in `gs_to` as columns.
	"""
	coords_from = shapely.get_coordinates(gs_from)
	coords_to = shapely.get_coordinates(gs_to)
	n_from = gs_from.size
	n_to = gs_to.size

	lngs, lats = numpy.repeat(coords_from, n_to, axis=0).T
	lngs2, lats2 = numpy.tile(coords_to, (n_from, 1)).T

	dist_func = haversine_distance if use_haversine else geod_distances
	distances = dist_func(lats, lngs, lats2, lngs2)
	return pandas.DataFrame(
		distances.reshape(n_from, n_to), index=gs_from.index, columns=gs_to.index
	)
