from collections.abc import Callable, Collection

import numpy
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point

from .util.distance import geod_distance_and_bearing, haversine_distance

PointSet = Collection[Point] | numpy.ndarray | GeoSeries | GeoDataFrame


def _geod_distance(lat: numpy.ndarray, lng: numpy.ndarray, target_lat: numpy.ndarray, target_lng: numpy.ndarray) -> numpy.ndarray:
	return geod_distance_and_bearing(lat, lng, target_lat, target_lng)[0]


def _to_lat_lngs(points: Collection[Point] | numpy.ndarray | GeoSeries):
	if isinstance(points, GeoSeries):
		lats = points.y.to_numpy()
		lngs = points.x.to_numpy()
	else:
		lats = numpy.asarray([point.y for point in points])
		lngs = numpy.asarray([point.x for point in points])
	return lats, lngs


def haversine_distances(points: PointSet, target: Point):
	if isinstance(points, GeoDataFrame):
		points = points.geometry
	lats, lngs = _to_lat_lngs(points)
	n = lats.size
	target_lat = numpy.repeat(target.y, n)
	target_lng = numpy.repeat(target.x, n)
	return haversine_distance(lats, lngs, target_lat, target_lng)


def geod_distances(points: PointSet, target: Point):
	if isinstance(points, GeoDataFrame):
		points = points.geometry
	lats, lngs = _to_lat_lngs(points)
	n = lats.size
	target_lat = numpy.repeat(target.y, n)
	target_lng = numpy.repeat(target.x, n)
	return geod_distance_and_bearing(lats, lngs, target_lat, target_lng)[0]


def _get_best_pic_inner(
	lats: numpy.ndarray,
	lngs: numpy.ndarray,
	target_lat: float,
	target_lng: float,
	dist_func: Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray],
) -> tuple[int, float]:
	target_lats = numpy.repeat(target_lat, lats.size)
	target_lngs = numpy.repeat(target_lng, lngs.size)
	distances = dist_func(lats, lngs, target_lats, target_lngs)
	index = numpy.argmin(distances)
	return index.item(), distances[index].item()


def get_best_pic(pics: PointSet, target: 'Point', *, use_haversine: bool = False):
	"""Finds the best pic among a collection of pics. If pics is a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index.

	Returns:
		tuple (index, distance in metres)"""
	if isinstance(pics, GeoDataFrame):
		pics = pics.geometry

	lats, lngs = _to_lat_lngs(pics)
	dist_func = haversine_distance if use_haversine else _geod_distance
	index, distance = _get_best_pic_inner(lats, lngs, target.y, target.x, dist_func)
	if isinstance(pics, GeoSeries):
		index = pics.index[index]
	return index, distance


def get_worst_point(pics: PointSet, targets: PointSet, *, use_haversine: bool = False):
	"""Finds the worst case distance in a group of targets, and the index of that target within `targets`. If `pics` or `targets` are a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""
	if isinstance(pics, GeoDataFrame):
		pics = pics.geometry
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	lats, lngs = _to_lat_lngs(pics)
	target_lats, target_lngs = _to_lat_lngs(targets)
	dist_func = haversine_distance if use_haversine else _geod_distance

	worst_dist = 0.0
	worst_pic = -1  # Misnomer, just shorter than saying "closest pic for worst target"
	worst_target = -1
	# Yeah I'm implementing max() manually basically
	# There would still definitely be a better way to implement this, but I'm not thinky that hard right now
	# Probably want to use a vectorized distance function here?
	for i, (target_lat, target_lng) in enumerate(zip(target_lats, target_lngs, strict=True)):
		pic_index, dist = _get_best_pic_inner(lats, lngs, target_lat, target_lng, dist_func)
		if dist > worst_dist:
			worst_dist = dist
			worst_target = i
			worst_pic = pic_index

	if isinstance(targets, GeoSeries):
		worst_target = targets.index[worst_target]
	if isinstance(pics, GeoSeries):
		worst_pic = pics.index[worst_pic]
	return worst_target, worst_dist, worst_pic
