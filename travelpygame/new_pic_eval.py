"""Tools to help find what adding new pics would do"""

import logging
from collections.abc import Collection, Hashable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas
import pandas
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point
from tqdm.auto import tqdm

from .best_pics import PointSet, get_best_pic
from .tpg_data import load_rounds
from .util.distance import get_distances
from .util.io_utils import load_points
from .util.kml import parse_submission_kml

if TYPE_CHECKING:
	import numpy

logger = logging.getLogger(__name__)


def _to_items(points: 'Collection[Point] | numpy.ndarray | GeoSeries'):
	if isinstance(points, GeoSeries):
		total = points.size
		items = points.items()
	else:
		total = len(points)
		items = enumerate(points)
	return total, items


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
	if isinstance(targets, Collection) and not isinstance(targets, (Sequence, GeoSeries)):
		targets = list(targets)

	total, items = _to_items(targets)
	new_point_distances = get_distances(new_point, targets, use_haversine=use_haversine)
	new_point_distance = pandas.Series(
		new_point_distances, index=targets.index if isinstance(targets, GeoSeries) else None
	)

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
			new_dist = new_point_distance[index]  # pyright: ignore[reportArgumentType, reportCallIssue]
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
	if isinstance(targets, GeoSeries):
		targets = targets.to_numpy()
	with tqdm(items, 'Finding impact of new points against targets', total) as t:
		for index, new_point in t:
			if not isinstance(new_point, Point):
				logger.warning(
					'new_points contained %s at index %s, expected Point', type(new_points), index
				)
				continue
			t.set_postfix(new_point=index)
			new_distances = get_distances(new_point, targets, use_haversine=use_haversine)
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
