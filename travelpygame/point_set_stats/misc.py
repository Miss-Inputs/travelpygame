from collections.abc import Collection
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas
import shapely
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm.auto import tqdm

from travelpygame.util.distance import geod_distance, get_distances
from travelpygame.util.geom_utils import get_bbox_corners

if TYPE_CHECKING:
	from geopandas import GeoSeries
	from numpy import ndarray


def _distance_metric_func(u: 'ndarray', v: 'ndarray', t: tqdm):
	t.update()
	return geod_distance((u[1], u[0]), (v[1], v[0]))


def find_clusters(
	points: 'GeoSeries',
	threshold: float,
	linkage_method: Literal['single', 'complete', 'average'] = 'average',
	*,
	use_tqdm: bool = True,
) -> pandas.Series[int]:
	"""Finds clusters using hierarchal clustering. May be slow on larger datasets.

	Arguments:
		points: GeoSeries containing points.
		threshold: Linkage threshold in metres.

	"""
	coords = shapely.get_coordinates(points)

	size = points.index.size
	# We can just use fclusterdata, but it's probably cleaner down the line to do it in two separate steps
	with tqdm(desc='Clustering', total=(size * (size - 1)) / 2, disable=not use_tqdm) as t:
		# Potentially, we want to use scipy.spatial.distance.pdist to get a distance matrix ourself, but eh, we can let linkage() do it
		linkage_matrix = linkage(coords, linkage_method, partial(_distance_metric_func, t=t))
		labels = fcluster(linkage_matrix, threshold, 'distance')
	return pandas.Series(labels, index=points.index)


def corner_distance(
	point: shapely.Point | tuple[float, float], *corners: shapely.Point, use_haversine: bool = False
) -> float:
	"""Sum of distances from `point` to each corner in `corners`, intended to be the corner points of a bounding box."""
	distances = get_distances(point, corners, use_haversine=use_haversine)
	return distances.sum().item()


def closest_to_corners(
	points: 'GeoSeries | Collection[shapely.Point]', *, use_haversine: bool = False
) -> shapely.Point:
	"""Returns the point in `points` that is closest to all corners of the bounding box of all of `points`, usually used to roughly determine a centre of the points."""
	corners = get_bbox_corners(points)
	return min(points, key=partial(corner_distance, *corners, use_haversine=use_haversine))
