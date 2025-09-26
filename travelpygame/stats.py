"""Various stats-related things where it didn't feel right putting them in util and I couldn't think of anywhere else to put them"""

import logging
from collections.abc import Collection

import numpy
import shapely
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from .util import geod_distance, get_antipode, haversine_distance

logger = logging.getLogger(__name__)


def _maximin_objective(x: numpy.ndarray, points: Collection[shapely.Point]) -> float:
	point = shapely.Point(x)
	return -min(geod_distance(point, p) for p in points)


def _maximin_haversine_objective(x: numpy.ndarray, points: Collection[shapely.Point]) -> float:
	lat, lng = x
	return -min(haversine_distance(lat, lng, p.y, p.x) for p in points)


def _find_furthest_point_single(points: Collection[shapely.Point]):
	point = next(iter(points))
	anti_lat, anti_lng = get_antipode(point.y, point.x)
	antipode = shapely.Point(anti_lng, anti_lat)
	# Can't be bothered remembering the _exact_ circumference of the earth, maybe I should to speed things up whoops
	return antipode, geod_distance(point, antipode)


def find_furthest_point_via_optimization(
	points: Collection[shapely.Point],
	initial: shapely.Point | None = None,
	max_iter: int = 1_000,
	pop_size: int = 20,
	*,
	use_tqdm: bool = True,
	use_haversine: bool = False,
) -> tuple[shapely.Point, float]:
	if len(points) == 1:
		return _find_furthest_point_single(points)
	# TODO: Should be able to trivially speed up len(points) == 2 by getting the midpoint of the two antipodes, unless I'm wrong
	bounds = ((-180, 180), (-90, 90))
	with tqdm(
		desc='Differentially evolving', total=(max_iter + 1) * pop_size * 2, disable=not use_tqdm
	) as t:
		# total should be actually (max_iter + 1) * popsize * 2 but eh I'll fiddle with that later
		def callback(*_):
			# If you just pass t.update to the callback= argument it'll just stop since t.update() returns True yippeeeee
			t.update()

		result = differential_evolution(
			_maximin_haversine_objective if use_haversine else _maximin_objective,
			bounds,
			popsize=pop_size,
			args=(points,),
			x0=numpy.asarray([initial.x, initial.y]) if initial else None,
			maxiter=max_iter,
			mutation=(0.5, 2.0),
			tol=1e-7,  # should probably be a argument
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
