"""Various stats-related things where it didn't feel right putting them in util and I couldn't think of anywhere else to put them"""

import logging
from collections.abc import Collection, Hashable

import numpy
import shapely
from geopandas import GeoDataFrame, GeoSeries
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from .util import geod_distance, get_closest_index, get_distances, get_geometry_antipode

logger = logging.getLogger(__name__)


def _maximin_objective(x: numpy.ndarray, *args) -> float:
	points = args[0]
	use_haversine = args[1] if len(args) > 1 else False
	polygon: shapely.Polygon | None = args[2] if len(args) > 2 else None

	lng, lat = x
	penalize = False
	if polygon and not shapely.contains_xy(polygon, lng, lat):
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
	with tqdm(points.items(), 'Getting uniqueness', points.size, unit='point') as t:
		for index, point in t:
			t.set_postfix(point=index)
			other = points.drop(index)
			if not isinstance(point, shapely.Point):
				raise TypeError(f'{index} was {type(point)}, expected Point')

			closest_index, dists[index] = get_closest_index(point, other.to_numpy())
			closest[index] = other.index[closest_index]
	return closest, dists


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
