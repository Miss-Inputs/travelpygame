from collections.abc import Collection
from typing import Any

import numpy
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point

from .util.distance import get_closest_index, get_furthest_index

PointSet = Collection[Point] | numpy.ndarray | GeoSeries | GeoDataFrame


def get_best_pic(
	pics: PointSet, target: 'Point', *, use_haversine: bool = False, reverse: bool = False
) -> tuple[Any, float]:
	"""Finds the best pic among a collection of pics. If pics is a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index.

	Arguments:
		reverse: If true, get the furthest away pic, for whatever reason.

	Returns:
		tuple (index, distance in metres)"""
	if isinstance(pics, GeoDataFrame):
		pics = pics.geometry

	if isinstance(pics, Collection) and not isinstance(pics, (GeoSeries, GeoDataFrame)):
		pics = list(pics)
	coords = shapely.get_coordinates(pics)
	index, distance = (
		get_furthest_index(target, coords, use_haversine=use_haversine)
		if reverse
		else get_closest_index(target, coords, use_haversine=use_haversine)
	)
	if isinstance(pics, GeoSeries):
		index = pics.index[index]
	return index, distance


def get_worst_point(pics: PointSet, targets: PointSet, *, use_haversine: bool = False):
	"""Finds the worst case distance in a group of targets, and the index of that target within `targets`. If `pics` or `targets` are a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""
	if isinstance(pics, GeoDataFrame):
		pics = pics.geometry
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	if isinstance(pics, Collection) and not isinstance(pics, GeoSeries):
		pics = list(pics)
	if isinstance(targets, Collection) and not isinstance(targets, GeoSeries):
		targets = list(targets)
	coords = shapely.get_coordinates(pics)

	worst_dist = 0.0
	worst_pic = -1  # Misnomer, just shorter than saying "closest pic for worst target"
	worst_target = -1
	# Yeah I'm implementing max() manually basically
	# There would still definitely be a better way to implement this, but I'm not thinky that hard right now
	# Probably want to use a vectorized distance function here?
	items = targets.items() if isinstance(targets, GeoSeries) else enumerate(targets)
	for target_index, target in items:
		if not isinstance(target, Point):
			raise TypeError(f'Target at {target_index} was {type(target)}, expected Point')
		pic_index, dist = get_closest_index(target, coords, use_haversine=use_haversine)
		if dist > worst_dist:
			worst_dist = dist
			worst_target = target_index
			worst_pic = pic_index

	if isinstance(pics, GeoSeries):
		worst_pic = pics.index[worst_pic]
	return worst_target, worst_dist, worst_pic
