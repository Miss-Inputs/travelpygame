"""Stuff that requires an optimization (in the mathematical sense)."""
import logging
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy
import shapely
from geopandas import GeoSeries
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from travelpygame.util.distance import geod_distance, get_distances
from travelpygame.util.geo_utils import get_geometry_antipode

if TYPE_CHECKING:
	from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

def _diagonal_dist(poly: 'BaseGeometry') -> float:
	minx, miny, maxx, maxy = poly.bounds
	return geod_distance((miny, minx), (maxy, maxx))


def _maximin_objective(x: numpy.ndarray, *args) -> float:
	points = args[0]
	use_haversine = args[1] if len(args) > 1 else False
	polygon: BaseGeometry | None = args[2] if len(args) > 2 else None

	lng, lat = x
	distances = get_distances((lat, lng), points, use_haversine=use_haversine)
	min_dist = distances.min()

	if polygon and not shapely.intersects_xy(polygon, lng, lat):
		# This doesn't always work as expected with multipolygons, like if polygon is a country with an offshore island, the optimizer tends to end up in the mainland and never the island even when it's visibly further away
		diagonal_dist = _diagonal_dist(polygon)
		return diagonal_dist - min_dist
	return -min_dist


def _geo_median_objective(x: numpy.ndarray, *args):
	"""Sum of distances to points."""
	points: Sequence[shapely.Point] | GeoSeries = args[0]
	use_haversine = args[1] if len(args) > 1 else False

	lng, lat = x
	distances = get_distances((lat, lng), points, use_haversine=use_haversine)
	return distances.sum()


def _find_furthest_point_single(points: Collection[shapely.Point]):
	point = next(iter(points))
	antipode = get_geometry_antipode(point)
	# Can't be bothered remembering the _exact_ circumference of the earth, maybe I should to speed things up whoops (I guess it's probably different for haversine vs geodetic?)
	return antipode, geod_distance(point, antipode)


def find_furthest_point(
	points: Collection[shapely.Point],
	initial: shapely.Point | None = None,
	polygon: 'BaseGeometry | None' = None,
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
		desc='Differentially evolving for furthest point',
		total=(max_iter + 1) * pop_size * 2,
		disable=not use_tqdm,
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
	if isinstance(distance, numpy.floating):
		# Those numpy floating types are probably going to bite me in the arse later if I don't stop them propagating
		distance = distance.item()
	return point, distance


def find_geometric_median(
	points: Sequence[shapely.Point] | GeoSeries,
	initial: shapely.Point | None = None,
	max_iter: int = 1_000,
	pop_size: int = 20,
	tolerance: float = 1e-7,
	*,
	use_tqdm: bool = True,
	use_haversine: bool = False,
) -> shapely.Point:
	if len(points) == 1:
		if isinstance(points, GeoSeries):
			first = points.iloc[0]
			if not isinstance(first, shapely.Point):
				raise TypeError(f'points contained single {type(first)} and not Point')
			return first
		return points[0]
	if isinstance(points, GeoSeries):
		minx, miny, maxx, maxy = points.total_bounds
	else:
		minx, miny, maxx, maxy = shapely.total_bounds(points)
	bounds = ((minx, maxx), (miny, maxy))
	with tqdm(
		desc='Differentially evolving for geometric median',
		total=(max_iter + 1) * pop_size * 2,
		disable=not use_tqdm,
	) as t:

		def callback(*_):
			# If you just pass t.update to the callback= argument it'll just stop since t.update() returns True yippeeeee
			t.update()

		result = differential_evolution(
			_geo_median_objective,
			bounds,
			popsize=pop_size,
			args=(points, use_haversine),
			x0=numpy.asarray([initial.x, initial.y]) if initial else None,
			maxiter=max_iter,
			mutation=(0.5, 1.5),
			tol=tolerance,
			callback=callback,
		)

	point = shapely.Point(result.x)
	if not result.success:
		logger.info(result.message)
	return point
