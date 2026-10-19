"""Functions related to getting a best pic within a point set to some point, maybe this could be merged into somewhere else."""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

from .util.distance import DistanceMethod, cartesian_product_distances

if TYPE_CHECKING:
	from shapely import Point

	from .point_set import PointSet


def get_best_pic(
	pics: 'PointSet',
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
	distances = pics.get_all_distances(target, distance_method)
	index = distances.idxmax() if reverse else distances.idxmin()
	distance = distances[index]  # ty: ignore[invalid-argument-type]q
	return index, distance


def get_worst_target(
	pics: 'PointSet', targets: 'PointSet', distance_method: DistanceMethod = DistanceMethod.Geodetic
) -> tuple[Hashable, float, Hashable]:
	"""Finds the worst case distance in a group of targets, and the index of that target within `targets`. If `pics` or `targets` are a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""

	distances = cartesian_product_distances(pics.points, targets.points, distance_method)
	unstacked = distances.unstack(sort=False)
	worst_target, worst_pic = unstacked.idxmax()  # ty: ignore[not-iterable] #it is a tuple
	worst_dist = unstacked.max().item()
	return worst_target, worst_dist, worst_pic
