"""Tools for measuring distance and such."""

from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from enum import StrEnum, auto
from itertools import combinations
from typing import overload

import numpy
import pandas
import pyproj
import shapely
from geopandas import GeoSeries
from numpy.typing import NDArray

from .geom_utils import get_poly_vertices

wgs84_geod = pyproj.Geod(ellps='WGS84')

type FloatNDArray = NDArray[numpy.floating]
type FloatListlike = Sequence[float] | FloatNDArray | pandas.Series
"""Accepted input types to pyproj.Geod.inv, although other stuff would probably work, this is just what works as a type hint."""


class DistanceMethod(StrEnum):
	"""Accepted distance calculation methods (not always all of them are supported by every function)"""

	Geodetic = auto()
	Haversine = auto()
	Euclidean = auto()
	Manhattan = auto()


EQUATORIAL_RADIUS_M = wgs84_geod.a  # 6_378_137.0
"""Earth radius at the equator in metres from WGS84."""
POLAR_RADIUS_M = wgs84_geod.b  # 6_356_752.314245179…
"""Earth radius from equator to poles in metres from WGS84."""
AVERAGE_RADIUS_M = (EQUATORIAL_RADIUS_M + POLAR_RADIUS_M) / 2
"""Average radius of the Earth in metres."""
_1deg_rad: float = numpy.radians(1).item()
LATITUDE_DEG_M = AVERAGE_RADIUS_M * _1deg_rad
"""Length of 1 degree of latitude in metres (longitude is variable, 111320 at the equator to 0m at the poles) (technically this is also variable but we will just say average earth radius * 1 degree in radians)."""
EQ_LONGITUDE_DEG_M = EQUATORIAL_RADIUS_M * _1deg_rad
"""Width of 1 degree longitude at the equator in metres (radius at equator * 1 degree in radians)."""
AVERAGE_LONGITUDE_DEG_M = EQ_LONGITUDE_DEG_M * (2 / numpy.pi)
"""Average width of longitudes in metres (equator width * average value of cosine curve)."""
AVERAGE_DEG_M: float = numpy.sqrt(((LATITUDE_DEG_M**2) + (AVERAGE_LONGITUDE_DEG_M**2)) / 2).item()
"""Average size of 1 degree in either direction in metres, which is of course nonsensical geographically speaking, as it pretends the Earth is a square."""


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
) -> tuple[FloatNDArray, FloatNDArray]: ...


def geod_distance_and_bearing(
	lat1: float | FloatListlike,
	lng1: float | FloatListlike,
	lat2: float | FloatListlike,
	lng2: float | FloatListlike,
	*,
	radians: bool = False,
) -> tuple[float | FloatNDArray, float | FloatNDArray]:
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
	lat1: FloatNDArray,
	lng1: FloatNDArray,
	lat2: FloatNDArray,
	lng2: FloatNDArray,
	*,
	radians: bool = False,
) -> FloatNDArray: ...


def haversine_distance(
	lat1: float | FloatNDArray,
	lng1: float | FloatNDArray,
	lat2: float | FloatNDArray,
	lng2: float | FloatNDArray,
	*,
	radians: bool = False,
) -> float | FloatNDArray:
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
	r = AVERAGE_RADIUS_M
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
	lat: FloatNDArray, lng: FloatNDArray, target_lat: FloatNDArray, target_lng: FloatNDArray
) -> FloatNDArray:
	"""Vectorized get_geod_distance_and_bearing that just gets the distance and not bearing (for symmetry with haversine_distance)."""
	return geod_distance_and_bearing(lat, lng, target_lat, target_lng)[0]


@overload
def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float: ...
@overload
def euclidean_distance(
	x1: FloatNDArray, y1: FloatNDArray, x2: FloatNDArray, y2: FloatNDArray
) -> FloatNDArray: ...
def euclidean_distance(
	x1: float | FloatNDArray,
	y1: float | FloatNDArray,
	x2: float | FloatNDArray,
	y2: float | FloatNDArray,
) -> float | FloatNDArray:
	"""Vectorized distance function for non-geographical coordinates."""
	return numpy.hypot(x1 - x2, y1 - y2)


@overload
def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float: ...
@overload
def manhattan_distance(
	x1: FloatNDArray, y1: FloatNDArray, x2: FloatNDArray, y2: FloatNDArray
) -> FloatNDArray: ...
def manhattan_distance(
	x1: float | FloatNDArray,
	y1: float | FloatNDArray,
	x2: float | FloatNDArray,
	y2: float | FloatNDArray,
) -> float | FloatNDArray:
	"""Vectorized distance function for non-geographical coordinates."""
	return numpy.abs(x1 - x2) + numpy.abs(y1 - y2)


def _vectorized_distance(
	lat1: FloatNDArray,
	lng1: FloatNDArray,
	lat2: FloatNDArray,
	lng2: FloatNDArray,
	method: DistanceMethod,
) -> FloatNDArray:
	if method == DistanceMethod.Geodetic:
		return geod_distance_and_bearing(lat1, lng1, lat2, lng2)[0]
	if method == DistanceMethod.Haversine:
		return haversine_distance(lat1, lng1, lat2, lng2)
	if method == DistanceMethod.Euclidean:
		return euclidean_distance(lng1, lat1, lng2, lat2) * AVERAGE_DEG_M
	if method == DistanceMethod.Manhattan:
		return manhattan_distance(lng1, lat1, lng2, lat2) * AVERAGE_DEG_M
	raise ValueError(f'Distance method {method} not understood')


def get_distances(
	target_point: shapely.Point | tuple[float, float],
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray | GeoSeries,
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> FloatNDArray:
	"""Finds the distances from all points in `points` to `target_point`, in the original order of points. By default, uses geodetic distance. If `target_point` is a tuple, it should be lat, lng. If points is a numpy array of floats, it must be 2D, wih one axis having size 2, noting that it expects lng/x first and not the other way around.

	Returns:
		1D numpy array of shape (len(points), ) containing distances in metres."""
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
		if isinstance(points, Collection) and not isinstance(points, (Sequence, GeoSeries)):
			points = list(points)
		lngs, lats = shapely.get_coordinates(points).T

	if isinstance(target_point, shapely.Point):
		target_lat = target_point.y
		target_lng = target_point.x
	else:
		target_lat, target_lng = target_point
	return _vectorized_distance(
		numpy.repeat(target_lat, lats.size),
		numpy.repeat(target_lng, lngs.size),
		lats,
		lngs,
		distance_method,
	)


def get_distance(
	point1: shapely.Point | tuple[float, float],
	point2: shapely.Point | tuple[float, float],
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> float:
	"""Scalar version of get_distances for completeness."""
	if distance_method == DistanceMethod.Geodetic:
		return geod_distance(point1, point2)

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

	if distance_method == DistanceMethod.Haversine:
		return haversine_distance(lat1, lng1, lat2, lng2)
	if distance_method == DistanceMethod.Euclidean:
		return euclidean_distance(lng1, lat1, lng2, lat2) * AVERAGE_DEG_M
	if distance_method == DistanceMethod.Manhattan:
		return manhattan_distance(lng1, lat1, lng2, lat2) * AVERAGE_DEG_M
	raise ValueError(f'Distance method {distance_method} not understood')


def get_closest_point(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint,
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> tuple[shapely.Point, float]:
	"""Finds the closest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns one of them.

	Returns:
		Point, distance in metres
	"""
	if isinstance(points, shapely.MultiPoint):
		points = list(points.geoms)
	if not isinstance(points, Sequence):
		points = list(points)
	distances = get_distances(target_point, points, distance_method)
	index = distances.argmin().item()
	return points[index], distances[index]


def get_closest_index(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray,
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> tuple[int, float]:
	"""Finds the index of the closest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns the index of one of them.

	Returns:
		Point, distance in metres
	"""
	distances = get_distances(target_point, points, distance_method)
	index = distances.argmin().item()
	return index, distances[index]


def get_furthest_index(
	target_point: shapely.Point,
	points: Collection[shapely.Point] | shapely.MultiPoint | numpy.ndarray,
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> tuple[int, float]:
	"""Finds the index of the furthest point and the distance to it in a collection of points. Uses geodetic distance by default. If multiple points are equally close, arbitrarily returns the index of one of them.

	Returns:
		Point, distance in metres
	"""
	distances = get_distances(target_point, points, distance_method)
	index = distances.argmax().item()
	return index, distances[index]


def get_closest_points(
	target_point: shapely.Point,
	points: 'Sequence[shapely.Point] | shapely.MultiPoint | numpy.ndarray',
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
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
	distances = _vectorized_distance(target_lat, target_lng, lats, lngs, distance_method)
	shortest = distances.min().item()
	return [point for i, point in enumerate(points) if distances[i] == shortest], shortest


def self_cartesian_product_distances(
	gs: GeoSeries, distance_method: DistanceMethod = DistanceMethod.Geodetic
) -> dict[Hashable, dict[Hashable, float]]:
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

	half_distances = _vectorized_distance(lats, lngs, lats2, lngs2, distance_method)
	for i, distance in enumerate(half_distances):
		from_i = gs.index[from_indexes[i]]
		to_i = gs.index[to_indexes[i]]
		distances[from_i][to_i] = distance.item()
		distances[to_i][from_i] = distance.item()
	return distances


def cartesian_product_distances(
	gs_from: GeoSeries, gs_to: GeoSeries, distance_method: DistanceMethod = DistanceMethod.Geodetic
) -> pandas.DataFrame:
	"""Distances from every point in `gs_from` to every point in `gs_to`. Tries to be as efficient as possible. Probably isn't.

	Results are undefined if either object has an index that is multi-level or not unique, it will probably just not work.

	Arguments:
		gs_from: GeoSeries, geometries must be points.
		gs_to: GeoSeries, geometries must be points.
		distance_method: DistanceMethod enum that chooses function to calculate distance. Can be geodetic (default), haversine (faster, inaccurate), or Euclidean (should be a tad faster but doesn't return metres and is also inaccurate even if it did, due to the Earth not being flat, but I guess it would work if gs_from/gs_to are not in WGS84).

	Returns:
		DataFrame with the index of `gs_from`, each row containing distances (in metres) to each point in `gs_to` as columns.
	"""
	coords_from = shapely.get_coordinates(gs_from)
	coords_to = shapely.get_coordinates(gs_to)
	n_from = gs_from.size
	n_to = gs_to.size

	lngs, lats = numpy.repeat(coords_from, n_to, axis=0).T
	lngs2, lats2 = numpy.tile(coords_to, (n_from, 1)).T

	distances = _vectorized_distance(lats, lngs, lats2, lngs2, distance_method)
	return pandas.DataFrame(
		distances.reshape(n_from, n_to), index=gs_from.index, columns=gs_to.index
	)


def get_point_to_polygon_distance(
	point: shapely.Point | tuple[float, float],
	polygon: shapely.Polygon | shapely.MultiPolygon,
	densification: float | None = 0.1,
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
) -> tuple[tuple[shapely.Point, float], tuple[shapely.Point, float]]:
	"""Returns roughly the closest and furthest point anywhere on `polygon` from `point`.
	Currently just looks at polygon vertices, so doesn't actually give the closest result possible, though the `densification` argument is there to segmentize the polygon boundaries into segments so you get more vertices. Either way this will be slow.
	May produce unexpected results if `point` is inside a hole inside `polygon` or something like that, for now.
	"""
	# Note that mathematically, the furthest point will be one of the vertices (apparently), so at least that part's easier
	# We still need to calcuate it this same way though… creating a second inner function just to not call argmin() seems pointless
	if densification:
		polygon = shapely.segmentize(polygon, densification)
	vertices = get_poly_vertices(polygon)
	distances = get_distances(point, vertices, distance_method)
	closest_index = distances.argmin().item()
	closest = vertices[closest_index]
	closest_dist = distances[closest_index].item()

	furthest_index = distances.argmax().item()
	furthest = vertices[furthest_index]
	furthest_dist = distances[furthest_index].item()
	return (closest, closest_dist), (furthest, furthest_dist)
