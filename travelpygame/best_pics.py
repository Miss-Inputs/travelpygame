"""Functions related to getting a best pic within a point set to some point, maybe this could be merged into somewhere else."""

from collections.abc import Collection, Hashable
from typing import Any

import numpy
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point

from .point_set import PointSet
from .util.distance import (
	DistanceMethod,
	cartesian_product_distances,
	get_closest_index,
	get_furthest_index,
)

type PointCollection = Collection[Point] | numpy.ndarray | GeoSeries | GeoDataFrame
"""Seems a bit odd for this to be defined here of all places? Oh well"""


def get_best_pic(
	pics: PointCollection | PointSet,
	target: 'Point',
	distance_method: DistanceMethod = DistanceMethod.Geodetic,
	*,
	reverse: bool = False,
) -> tuple[Any, float]:
	"""Finds the best pic among a collection of pics. If pics is a GeoDataFrame/GeoSeries or PointSet, returns the index in that object and not the numeric index.

	Arguments:
		reverse: If true, get the furthest away pic, for whatever reason.

	Returns:
		tuple (index, distance in metres)"""
	if isinstance(pics, PointSet):
		coords = pics.coord_array
	else:
		if isinstance(pics, GeoDataFrame):
			pics = pics.geometry

		if isinstance(pics, Collection) and not isinstance(pics, (GeoSeries, GeoDataFrame)):
			pics = list(pics)
		coords = shapely.get_coordinates(pics)
	index, distance = (
		get_furthest_index(target, coords, distance_method)
		if reverse
		else get_closest_index(target, coords, distance_method)
	)

	if isinstance(pics, GeoSeries):
		index = pics.index[index]
	elif isinstance(pics, PointSet):
		index = pics[index]
	return index, distance


def get_worst_target(
	pics: PointSet, targets: PointSet, distance_method: DistanceMethod = DistanceMethod.Geodetic
) -> tuple[Hashable, float, Hashable]:
	"""Finds the worst case distance in a group of targets, and the index of that target within `targets`. If `pics` or `targets` are a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""

	distances = cartesian_product_distances(pics.points, targets.points, distance_method)
	unstacked = distances.unstack(sort=False)
	worst_target, worst_pic = unstacked.idxmax()  # ty: ignore[not-iterable] #it is a tuple
	worst_dist = unstacked.max().item()
	return worst_target, worst_dist, worst_pic
