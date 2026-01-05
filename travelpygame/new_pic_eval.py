"""Tools to help find what adding new pics would do"""

import logging
from bisect import bisect
from collections.abc import Collection, Hashable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas
import pandas
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point
from tqdm.auto import tqdm

from .best_pics import PointCollection, get_best_pic
from .submission_comparison import compare_player_in_round
from .tpg_data import Round, load_rounds
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
			[{'name': r.name, 'geometry': r.target} for r in rounds],
			geometry='geometry',
			crs='wgs84',
		)
	return load_points(path).dropna(subset='geometry')


def load_points_or_rounds(paths: Path | Sequence[Path]) -> GeoDataFrame:
	"""Simply loads either points from a spreadsheet/csv/geojson/etc file as with load_points, or a submission tracker if it is KMZ or KML. Does not do anything involving the existing submissions."""
	if isinstance(paths, Path):
		return _load_points_or_rounds_single(paths)
	points = [_load_points_or_rounds_single(path) for path in paths]
	gdf = pandas.concat(points)
	assert isinstance(gdf, GeoDataFrame)
	return gdf


def find_if_new_pics_better(
	points: PointCollection, new_points: PointCollection, targets: PointCollection, *, use_haversine: bool = False
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


def find_new_pic_diffs(
	points: PointCollection,
	new_point: Point,
	targets: PointCollection,
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
	points: PointCollection, new_points: PointCollection, targets: PointCollection, *, use_haversine: bool = False
) -> pandas.DataFrame:
	"""For each new point in `new_points`: Finds how often that new point was closer to a point in `targets` compared to `points`, and the total reduction in distance. This function's name kinda sucks, and it is also a tad convoluted and its purpose is also a bit murky, so it may be rewritten mercilessly or removed in future."""
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


@dataclass
class DistanceImprovement:
	"""Represents an instance of a new pic improving your distance to some kind of target."""

	target_name: str | None
	"""Name of the target, round, etc. if applicable."""
	target: Point
	"""Location of the target."""
	old_location_name: str | None
	"""Name of previous location that was closest, if applicable."""
	old_location: Point
	old_distance: float
	new_location_name: str | None
	"""Name of new location that would be closest, if applicable."""
	new_location: Point
	"""New location to be evaluated that is closer than old_location."""
	new_distance: float

	@property
	def amount(self):
		return self.old_distance - self.new_distance


def find_improvements_in_round(
	round_: Round,
	player_name: str,
	new_pics: PointCollection,
	distance_required: float | None = None,
	*,
	use_haversine: bool = True,
) -> Iterator[DistanceImprovement]:
	"""Finds where a previous round could have been improved by at least one place if any of new_pics was available at the time.

	Arguments:
		round_: Rounds from TPG data, preferably already scored (or had distance calculated).
		player_name: Player's name, if None, results might not entirely make sense (it would return something like every time anyone's submission at all gets improved by something in new_pics), but it's technically possible to do that.
		new_points: Set of hypothetical new locations to evaluate.
		distance_required: Only count if it is above this distance.
		use_haversine: If true, use haversine distance to calculate the distances from new_pics, as well as the existing round distances if it has not been scored, otherwise use WGS84 geodetic distance.
	"""

	if isinstance(new_pics, GeoDataFrame):
		new_pics = new_pics.geometry
	if isinstance(new_pics, Collection) and not isinstance(new_pics, (Sequence, GeoSeries)):
		new_pics = list(new_pics)
	submission_diff = compare_player_in_round(round_, player_name, use_haversine=use_haversine)
	if submission_diff is None:
		return
	new_distances = get_distances(submission_diff.target, new_pics, use_haversine=use_haversine)
	for i in range(len(new_distances)):
		new_distance = new_distances[i]
		if new_distance >= submission_diff.rival_distance:
			continue
		if distance_required is not None and new_distance >= distance_required:
			continue
		new_name = new_pics.index[i] if isinstance(new_pics, GeoSeries) else None
		new_loc = new_pics.iloc[i] if isinstance(new_pics, GeoSeries) else new_pics[i]
		assert isinstance(new_loc, Point), f'new_loc was {type(new_loc)}, expected Point'
		yield DistanceImprovement(
			submission_diff.round_name,
			submission_diff.target,
			None,
			submission_diff.player_pic,
			submission_diff.player_distance,
			new_name if isinstance(new_name, str) else None,
			new_loc,
			new_distance,
		)


def find_improvements_in_rounds(
	rounds: list[Round], player_name: str, new_pics: PointCollection, *, use_haversine: bool = True
) -> Iterator[DistanceImprovement]:
	"""Finds where previous rounds could have been improved by at least one place if any of new_pics was available at the time.

	Arguments:
		rounds: Rounds from TPG data, preferably already scored (or had distance calculated).
		player_name: Player's name
		new_points: Set of hypothetical new locations to evaluate.
		distance_required: Only count if it is above this distance.
	"""
	if isinstance(new_pics, GeoDataFrame):
		new_pics = new_pics.geometry
	if isinstance(new_pics, Collection) and not isinstance(new_pics, (Sequence, GeoSeries)):
		new_pics = list(new_pics)

	for round_ in rounds:
		yield from find_improvements_in_round(
			round_, player_name, new_pics, use_haversine=use_haversine
		)


def new_distance_rank(distance: float, round_: Round) -> int:
	"""Finds what ranking a disttance would get in a round (if it was based purely on distance). Assumes round is already scored!

	Returns:
		New ranking, which is effectively a 1-based index for submissions sorted by distance
	"""
	distances = [sub.distance for sub in round_.submissions if sub.distance is not None]
	if not distances:
		return 1
	distances.sort()
	return bisect(distances, distance) + 1
