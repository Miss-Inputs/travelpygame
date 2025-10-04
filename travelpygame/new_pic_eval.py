"""Tools to help find what adding new pics would do"""

import logging
from collections.abc import Callable, Collection, Hashable, Sequence
from operator import itemgetter
from pathlib import Path

import geopandas
import numpy
import pandas
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point
from tqdm.auto import tqdm

from .tpg_data import load_rounds
from .util.distance import geod_distance, geod_distance_and_bearing, haversine_distance
from .util.io_utils import load_points
from .util.kml import parse_submission_kml

logger = logging.getLogger(__name__)

PointSet = Collection[Point] | numpy.ndarray | GeoSeries | GeoDataFrame


def _geod_distance(lat: float, lng: float, target_lat: float, target_lng: float) -> float:
	return geod_distance_and_bearing(lat, lng, target_lat, target_lng)[0]


def _to_lat_lngs(points: Collection[Point] | numpy.ndarray | GeoSeries):
	if isinstance(points, GeoSeries):
		lats = points.y.to_numpy()
		lngs = points.x.to_numpy()
	else:
		lats = numpy.asarray(point.y for point in points)
		lngs = numpy.asarray(point.x for point in points)
	return lats, lngs


def _to_items(points: Collection[Point] | numpy.ndarray | GeoSeries):
	if isinstance(points, GeoSeries):
		total = points.size
		items = points.items()
	else:
		total = len(points)
		items = enumerate(points)
	return total, items


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
	dist_func: Callable[[float, float, float, float], float],
) -> tuple[int, float]:
	"""Uses a generator, which I _think_ might be faster than using vectorized functions to compute all distances to target_lat/target_lng at  once, but I'm not sure and haven't bothered benchmarking or anything"""
	generator = (
		(i, dist_func(lat, lng, target_lat, target_lng))
		for i, (lat, lng) in enumerate(iterable=zip(lats, lngs, strict=True))
	)
	return min(generator, key=itemgetter(1))


def get_best_pic(pics: PointSet, target: 'Point', *, use_haversine: bool = False):
	"""Finds the best pic among a collection of pics. If pics is a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""
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


def _load_points_or_rounds_single(path: Path):
	ext = path.suffix[1:].lower()
	if ext in {'kml', 'kmz'}:
		# It is assumed to be something exported from the submission tracker
		tracker = parse_submission_kml(path)
		return geopandas.GeoDataFrame(
			[{'name': r.name, 'geometry': r.target} for r in tracker.rounds],
			geometry='geometry',
			crs='wgs84',
		)
	if ext == 'json':
		rounds = load_rounds(path)
		return geopandas.GeoDataFrame(
			[{'name': r.name, 'geometry': Point(r.longitude, r.latitude)} for r in rounds],
			geometry='geometry',
			crs='wgs84',
		)
	return load_points(path)


def load_points_or_rounds(paths: Path | Sequence[Path]) -> GeoDataFrame:
	"""Simply loads either points from a spreadsheet/csv/geojson/etc file as with load_points, or a submission tracker if it is KMZ or KML. Does not do anything involving the existing submissions."""
	if isinstance(paths, Path):
		return _load_points_or_rounds_single(paths)
	points = [_load_points_or_rounds_single(path) for path in paths]
	gdf = pandas.concat(points)
	assert isinstance(gdf, GeoDataFrame)
	return gdf


def find_if_new_pics_better(
	points: PointSet, new_points: PointSet, targets: PointSet, *, use_haversine: bool = False
) -> pandas.DataFrame:
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	total, items = _to_items(targets)
	results = {}
	with tqdm(items, 'Finding if new pics are better', total) as t:
		for index, target in t:
			if not isinstance(target, Point):
				logger.warning(
					'targets contained %s at index %s, expected Point', type(target), index
				)
				continue
			t.set_postfix(index=index)
			point, distance = get_best_pic(points, target, use_haversine=use_haversine)
			new_point, new_distance = get_best_pic(new_points, target, use_haversine=use_haversine)
			result = {
				'current_best': point,
				'current_distance': distance,
				'new_best': new_point,
				'new_distance': new_distance,
				'is_new_better': distance > new_distance,
			}
			results[index] = result
	return pandas.DataFrame.from_dict(results, 'index')


def find_new_pic_diff(
	points: PointSet,
	new_point: Point,
	targets: PointSet,
	*,
	use_haversine: bool = False,
	use_tqdm: bool = True,
) -> pandas.DataFrame:
	"""Finds the differences in the best distances from a set of points to targets, and a new point to each target, and whether that is better."""
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	total, items = _to_items(targets)
	# TODO: Whoops I should be using vectorized distance functions for new_point -> targets
	results = {}
	with tqdm(
		items, 'Finding distances for current points and new point', total, disable=not use_tqdm
	) as t:
		for index, target in t:
			if not isinstance(target, Point):
				logger.warning(
					'targets contained %s at index %s, expected Point', type(target), index
				)
				continue
			t.set_postfix(index=index)
			point, distance = get_best_pic(points, target, use_haversine=use_haversine)
			new_dist = (
				haversine_distance(new_point.y, new_point.x, target.y, target.x)
				if use_haversine
				else geod_distance(new_point, target)
			)
			diff = distance - new_dist
			results[index] = {
				'current_best': point,
				'current_distance': distance,
				'new_distance': new_dist,
				'diff': diff,
				'is_better': new_dist < distance,
			}
	return pandas.DataFrame.from_dict(results, 'index')


def find_new_pics_better_individually(
	points: PointSet, new_points: PointSet, targets: PointSet, *, use_haversine: bool = False
) -> pandas.DataFrame:
	"""For each new point in `new_points`: Finds how often that new point was closer to a point in `targets` compared to `points`, and the total reduction in distance. This function's name kinda sucks."""
	if isinstance(new_points, GeoDataFrame):
		new_points = new_points.geometry
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	total, items = _to_items(targets)
	current_distances_d: dict[Hashable, float] = {}
	with tqdm(items, 'Finding current best distances', total) as t:
		for index, target in t:
			if not isinstance(target, Point):
				logger.warning(
					'targets contained %s at index %s, expected Point', type(new_points), index
				)
				continue
			t.set_postfix(target=index)
			current_distances_d[index] = get_best_pic(points, target, use_haversine=use_haversine)[
				1
			]
	current_distances = pandas.Series(current_distances_d)

	results = {}
	total, items = _to_items(new_points)
	with tqdm(items, 'Finding impact of new points against targets', total) as t:
		for index, new_point in t:
			if not isinstance(new_point, Point):
				logger.warning(
					'new_points contained %s at index %s, expected Point', type(new_points), index
				)
				continue
			t.set_postfix(new_point=index)
			new_distances = haversine_distances(targets, new_point)
			is_better = new_distances < current_distances
			if not is_better.any():
				continue
			diffs = current_distances - new_distances
			improvements = diffs[is_better]

			total_diff = improvements.sum()
			best = improvements.idxmax()
			results[index] = {
				'num_targets_better': is_better.sum(),
				'total': total_diff,
				'best': improvements.loc[best],
				'most_improved': best,
				'mean': improvements.mean(),
			}
	return pandas.DataFrame.from_dict(results, 'index')
