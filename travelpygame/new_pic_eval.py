"""Tools to help find what adding new pics would do"""

import logging
from collections.abc import Collection, Sequence
from operator import itemgetter
from pathlib import Path

import geopandas
import numpy
import pandas
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point
from tqdm.auto import tqdm

from .util.distance import geod_distance_and_bearing, haversine_distance
from .util.io_utils import load_points
from .util.kml import parse_submission_kml

logger = logging.getLogger(__name__)


def _geod_distance(lat: float, lng: float, target_lat: float, target_lng: float):
	return geod_distance_and_bearing(lat, lng, target_lat, target_lng)[0]


PointSet = Collection['Point'] | numpy.ndarray | GeoSeries | GeoDataFrame


def get_best_pic(pics: PointSet, target: 'Point', *, use_haversine: bool = False):
	"""Finds the best pic among a collection of pics. If pics is a GeoDataFrame/GeoSeries, returns the index in that object and not the numeric index."""
	if isinstance(pics, GeoDataFrame):
		pics = pics.geometry

	if isinstance(pics, GeoSeries):
		lats = pics.y.to_numpy()
		lngs = pics.x.to_numpy()
	else:
		lats = numpy.asarray(point.y for point in pics)
		lngs = numpy.asarray(point.x for point in pics)
	dist_func = haversine_distance if use_haversine else _geod_distance
	generator = (
		(i, dist_func(lat, lng, target.y, target.x))
		for i, (lat, lng) in enumerate(zip(lats, lngs, strict=True))
	)
	index, distance = min(generator, key=itemgetter(1))
	if isinstance(pics, GeoSeries):
		index = pics.index[index]
	return index, distance


def _load_points_or_rounds_single(path: Path):
	if path.suffix[1:].lower() in {'kml', 'kmz'}:
		# It is assumed to be something exported from the submission tracker
		tracker = parse_submission_kml(path)
		return geopandas.GeoDataFrame(
			[{'name': r.name, 'geometry': r.target} for r in tracker.rounds],
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
):
	if isinstance(targets, GeoDataFrame):
		targets = targets.geometry

	if isinstance(targets, GeoSeries):
		total = targets.size
		items = targets.items()
	else:
		total = len(targets)
		items = enumerate(targets)
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
