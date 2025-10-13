"""Various stats-related things for point sets"""

import logging
from collections.abc import Callable, Collection, Hashable
from enum import Enum, auto
from itertools import product
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import numpy
import pandas
import shapely
from geopandas import GeoDataFrame, GeoSeries
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from .util import geod_distance, get_closest_index, get_distances, get_geometry_antipode

if TYPE_CHECKING:
	from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def _maximin_objective(x: numpy.ndarray, *args) -> float:
	points = args[0]
	use_haversine = args[1] if len(args) > 1 else False
	polygon: shapely.Polygon | None = args[2] if len(args) > 2 else None

	lng, lat = x
	penalize = False
	if polygon and not shapely.intersects_xy(polygon, lng, lat):
		penalize = True

	distances = get_distances((lat, lng), points, use_haversine=use_haversine)
	min_dist = distances.min()
	return min_dist if penalize else -min_dist


def _find_furthest_point_single(points: Collection[shapely.Point]):
	point = next(iter(points))
	antipode = get_geometry_antipode(point)
	# Can't be bothered remembering the _exact_ circumference of the earth, maybe I should to speed things up whoops (I guess it's probably different for haversine vs geodetic?)
	return antipode, geod_distance(point, antipode)


def find_furthest_point(
	points: Collection[shapely.Point],
	initial: shapely.Point | None = None,
	polygon: shapely.Polygon | shapely.MultiPolygon | None = None,
	max_iter: int = 1_000,
	pop_size: int = 20,
	tolerance: float = 1e-7,
	*,
	use_tqdm: bool = True,
	use_haversine: bool = False,
) -> tuple[shapely.Point, float]:
	if len(points) == 1 and not polygon:
		return _find_furthest_point_single(points)
	# TODO: Should be able to trivially speed up len(points) == 2 by getting the midpoint of the two antipodes, unless I'm wrong
	if polygon:
		minx, miny, maxx, maxy = polygon.bounds
		bounds = ((minx, maxx), (miny, maxy))
		shapely.prepare(polygon)
	else:
		bounds = ((-180, 180), (-90, 90))
	with tqdm(
		desc='Differentially evolving', total=(max_iter + 1) * pop_size * 2, disable=not use_tqdm
	) as t:

		def callback(*_):
			# If you just pass t.update to the callback= argument it'll just stop since t.update() returns True yippeeeee
			t.update()

		result = differential_evolution(
			_maximin_objective,
			bounds,
			popsize=pop_size,
			args=(points, use_haversine, polygon),
			x0=numpy.asarray([initial.x, initial.y]) if initial else None,
			maxiter=max_iter,
			mutation=(0.5, 1.5),
			tol=tolerance,
			callback=callback,
		)

	point = shapely.Point(result.x)
	distance = -result.fun
	if not result.success:
		logger.info(result.message)
	if not isinstance(distance, float):
		# Those numpy floating types are probably going to bite me in the arse later if I don't stop them propagating
		distance = float(distance)
	return point, distance


def get_uniqueness(points: GeoSeries | GeoDataFrame):
	"""Finds the distance of each point to the closest other point."""
	if isinstance(points, GeoDataFrame):
		points = points.geometry

	closest: dict[Hashable, Hashable] = {}
	dists: dict[Hashable, float] = {}
	with tqdm(points.items(), 'Getting uniqueness', points.size, unit='point', leave=False) as t:
		for index, point in t:
			t.set_postfix(point=index)
			other = points.drop(index)
			if not isinstance(point, shapely.Point):
				raise TypeError(f'{index} was {type(point)}, expected Point')

			closest_index, dists[index] = get_closest_index(point, other.to_numpy())
			closest[index] = other.index[closest_index]
	return closest, dists


def get_total_uniqueness(points: GeoSeries | GeoDataFrame):
	"""Finds the total distance of each point to all other points."""
	if isinstance(points, GeoDataFrame):
		points = points.geometry

	total_dists: dict[Hashable, float] = {}
	with tqdm(
		points.items(), 'Getting total uniqueness', points.size, unit='point', leave=False
	) as t:
		for index, point in t:
			t.set_postfix(point=index)
			other = points.drop(index)
			if not isinstance(point, shapely.Point):
				raise TypeError(f'{index} was {type(point)}, expected Point')

			distances = get_distances(point, other.to_numpy())
			total_dists[index] = distances.sum().item()
	total_uniqueness = pandas.Series(total_dists, name='total_uniqueness')
	return total_uniqueness.sort_values(ascending=False)


def get_uniqueness_with_groups(points: GeoDataFrame, col_name: str):
	"""Finds the distance of each point to the closest other point out of points that do not have the same value for `col_name` as that point."""
	closest: dict[Hashable, Hashable] = {}
	dists: dict[Hashable, float] = {}
	with tqdm(points.iterrows(), 'Getting uniqueness', points.size, unit='point') as t:
		for index, row in t:
			t.set_postfix(point=index)
			other = points[points[col_name] != row[col_name]]
			point = row.geometry
			if not isinstance(point, shapely.Point):
				raise TypeError(f'{index} was {type(point)}, expected Point')

			closest_index, dists[index] = get_closest_index(point, other.geometry.to_numpy())
			closest[index] = other.index[closest_index]
	return closest, dists


class Distance1ToManyMethod(Enum):
	"""How to aggregate the distances from point_a to each point in points_b."""

	Mean = auto()
	SquaredMean = auto()
	Median = auto()
	Min = auto()
	Max = auto()
	Sum = auto()
	SquaredSum = auto()


def _point_set_distance_inner(
	point_a: 'BaseGeometry', points_b: GeoSeries, method: Distance1ToManyMethod
) -> tuple[float, float, Any]:
	"""Returns some aggregation of distances from point_a to points in points_b according to method, but also closest distance and index/name of closest point in points_b."""
	if not isinstance(point_a, shapely.Point):
		raise TypeError(f'point_a was {type(point_a)}, expected Point')
	distances = get_distances(point_a, points_b)
	min_index = distances.argmin().item()
	min_dist = distances[min_index]
	if method == Distance1ToManyMethod.Mean:
		score = distances.mean().item()
	elif method == Distance1ToManyMethod.Median:
		score = numpy.median(distances).item()
	elif method == Distance1ToManyMethod.Min:
		score = min_dist
	elif method == Distance1ToManyMethod.Max:
		score = distances.max().item()
	elif method == Distance1ToManyMethod.Sum:
		score = distances.sum().item()
	elif method == Distance1ToManyMethod.SquaredMean:
		score = numpy.square(distances.mean()).item()
	elif method == Distance1ToManyMethod.SquaredSum:
		score = numpy.square(distances.sum()).item()
	return score, min_dist, points_b.index[min_index]


class DistanceAggMethod(Enum):
	"""How to aggregate the scores from _point_set_distance_inner of each point_a."""

	Mean = auto()
	Median = auto()
	Min = auto()
	Max = auto()
	Sum = auto()


class PointSetDistanceMethod(Enum):
	"""Different methods of stipulating distance from one point set to another. Not all are symmetrical."""

	Hausdorff = (DistanceAggMethod.Max, Distance1ToManyMethod.Min)
	"""Max of every point in set A's closest distance to any point in set B."""
	MeanMin = (DistanceAggMethod.Mean, Distance1ToManyMethod.Min)
	"""Mean of every point in set A's closest distance to any point in set B."""
	MinMin = (DistanceAggMethod.Min, Distance1ToManyMethod.Min)
	"""Min of every point in set A's closest distance to any point in set B."""

	MeanMean = (DistanceAggMethod.Mean, Distance1ToManyMethod.Mean)
	"""Mean of every point in set A's mean of distances to points in set B."""
	TotalMean = (DistanceAggMethod.Sum, Distance1ToManyMethod.Mean)
	"""Sum of every point in set A's mean of distances to points in set B."""
	MeanTotal = (DistanceAggMethod.Mean, Distance1ToManyMethod.Sum)
	"""Mean of every point in set A's sum of distances to points in set B."""
	MeanSquaredError = (DistanceAggMethod.Mean, Distance1ToManyMethod.SquaredSum)
	"""Mean of every point in set A's squared sum of distances to points in set B."""
	MeanMedian = (DistanceAggMethod.Mean, Distance1ToManyMethod.Median)
	"""Mean of every point in set A's median of distances to points in set B."""
	MedianMedian = (DistanceAggMethod.Median, Distance1ToManyMethod.Median)
	"""Median of every point in set A's median of distances to points in set B."""

	# And so on, and so forth. There's not any real reason to list them all here unless you want to give them special names, or for the sake of having some presets


DistanceAggMethodType = DistanceAggMethod | Callable[[Collection[float]], float]
CustomPointSetDistanceMethod = tuple[DistanceAggMethodType, Distance1ToManyMethod]
PointSetDistanceMethodType = PointSetDistanceMethod | CustomPointSetDistanceMethod


def get_distance_method_combinations(
	*, one_name_per_method: bool = True
) -> dict[str, CustomPointSetDistanceMethod]:
	"""Useful for CLIs, etc."""
	combos = product(DistanceAggMethod, Distance1ToManyMethod)
	if one_name_per_method:
		return {
			f'{agg_method.name.lower()}_{inner_method.name.lower()}': (agg_method, inner_method)
			for agg_method, inner_method in combos
		}
	d: dict[str, CustomPointSetDistanceMethod] = {}
	for agg_method, inner_method in combos:
		name = f'{agg_method.name}{inner_method.name}'
		name_underscore = f'{agg_method.name}_{inner_method.name}'
		d.update(
			(name, (agg_method, inner_method))
			for name in {name, name_underscore, name.lower(), name_underscore.lower()}
		)
	return d


def get_point_set_distance(
	points_a: GeoSeries,
	points_b: GeoSeries,
	method: PointSetDistanceMethodType = PointSetDistanceMethod.MeanMean,
	*,
	use_tqdm: bool = True,
) -> tuple[float, float, str, str]:
	"""Calculates distance/difference/dissimilarity from points_a to point_b in a variety of different ways. Should be symmetrical.

	Returns:
		Tuple: dissimilarity (float, higher = less similar), distance between closest points, index/name of closest point in A to any point in B, index/name of closest point in B to any point in A
	"""
	if isinstance(method, tuple):
		outer_method, inner_method = method
	else:
		outer_method, inner_method = method.value

	scores: list[float] = []
	closest_distances: list[tuple[float, str, str]] = []

	with tqdm(
		desc=f'Finding point set distance between {points_a.name} and {points_b.name}',
		total=points_a.size + points_b.size,
		unit='point',
		disable=not use_tqdm,
	) as t:
		for index_a, point_a in points_a.items():
			t.update()
			index_a = str(index_a)
			score, closest_dist_b, index_b = _point_set_distance_inner(
				point_a, points_b, inner_method
			)
			scores.append(score)
			closest_distances.append((closest_dist_b, index_a, str(index_b)))
		for point_b in points_b:
			t.update()
			# Do it again to ensure symmetry
			score = _point_set_distance_inner(point_b, points_a, inner_method)[0]
			scores.append(score)

	if outer_method == DistanceAggMethod.Max:
		dist = max(scores)
	elif outer_method == DistanceAggMethod.Mean:
		dist = numpy.mean(scores).item()
	elif outer_method == DistanceAggMethod.Min:
		dist = min(scores)
	elif outer_method == DistanceAggMethod.Sum:
		dist = sum(scores)
	elif outer_method == DistanceAggMethod.Median:
		dist = numpy.median(scores).item()
	elif callable(outer_method):
		dist = outer_method(scores)
	else:
		raise ValueError(f'Unknown distance aggregation method: {(outer_method)}')

	closest_dist, closest_a, closest_b = min(closest_distances, key=itemgetter(0))
	return dist, closest_dist, closest_a, closest_b


def validate_points(
	geo: GeoSeries | GeoDataFrame, rounding_tolerance: int | None = 6, name_for_log: Any = None
) -> GeoSeries:
	"""Validates a point set to check it has no NaN/infinity coordinates, and removes duplicates (by default, considering points the same if their coordinates rounded down to 6 decimal places are the same). Does not necessarily validate that anything is not a point (at least not properly), you should do that yourself."""
	if isinstance(geo, GeoDataFrame):
		geo = geo.geometry

	coords = shapely.get_coordinates(geo)
	if rounding_tolerance is not None:
		coords = coords.round(rounding_tolerance)
	first_points: dict[tuple[float, float], tuple[Hashable, shapely.Point]] = {}
	"""The first instance of every point. Doesn't really need to be a dict I guess."""
	to_drop = set()

	for i, (index, item) in enumerate(geo.items()):
		x, y = coords[i]

		checks = (
			(numpy.isnan(x), 'NaN longitude'),
			(numpy.isnan(y), 'NaN latitude'),
			(numpy.isinf(x), 'infinity longitude'),
			(numpy.isinf(y), 'infinity latitude'),
			(x > 180, 'longitude too east'),
			(x < -180, 'longitude too west'),
			(y > 90, 'latitude too north'),
			(y < -90, 'latitude too south'),
		)

		coords_valid = True
		for check, desc in checks:
			if check:
				logger.info('%s had point %s with %s', name_for_log, index, desc)
				coords_valid = False
				break

		if coords_valid:
			if (x, y) in first_points:
				logger.info(
					'%s had duplicate point %s (identical to %s, %s)',
					name_for_log,
					index,
					*first_points[x, y],
				)
				to_drop.add(index)
			else:
				assert isinstance(item, shapely.Point), (
					f'uh oh item at {index} is actually {type(item)}'
				)
				first_points[x, y] = index, item
		else:
			to_drop.add(index)
	return geo.drop(list(to_drop)) if to_drop else geo
