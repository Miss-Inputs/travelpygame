from contextlib import nullcontext

import geopandas
import numpy
import shapely
from tqdm.auto import tqdm

from .util.geo_utils import contains_any, contains_any_array

RandomSeed = (
	numpy.random.Generator | numpy.random.BitGenerator | numpy.random.SeedSequence | int | None
)
"""Acceptable input types for a random seed."""


def random_point_in_bbox(
	min_x: float, min_y: float, max_x: float, max_y: float, random: RandomSeed = None
) -> shapely.Point:
	"""Uniformly generates a point somewhere in a bounding box."""
	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)
	x = random.uniform(min_x, max_x)
	y = random.uniform(min_y, max_y)
	return shapely.Point(x, y)

def random_points_in_bbox(
	n: int, min_x: float, min_y: float, max_x: float, max_y: float, random: RandomSeed = None
):
	"""Uniformly generates points somewhere in a bounding box."""
	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)
	x = random.uniform(min_x, max_x, n)
	y = random.uniform(min_y, max_y, n)
	points = shapely.points(x, y)
	assert not isinstance(points, shapely.Point)
	return points

def random_point_in_poly(
	poly: shapely.Polygon | shapely.MultiPolygon | geopandas.GeoSeries | geopandas.GeoDataFrame,
	random: RandomSeed = None,
	*,
	use_tqdm: bool = False,
	**tqdm_kwargs,
) -> shapely.Point:
	"""
	Uniformly-ish generates a point somewhere within a polygon.
	This won't choose anywhere directly on the edge (I think). If poly is a MultiPolygon, it will be inside one of the components, but the distribution of which one might not necesarily be uniform.

	Arguments:
		poly: shapely Polygon or MultiPolygon
		random: Optionally a numpy random generator or seed, otherwise default_rng is used
	"""
	if isinstance(poly, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
		min_x, min_y, max_x, max_y = poly.total_bounds
	else:
		min_x, min_y, max_x, max_y = poly.bounds
		shapely.prepare(poly)

	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)

	t = tqdm(**tqdm_kwargs) if use_tqdm else nullcontext()
	with t:
		while True:
			if isinstance(t, tqdm):
				t.update(1)
			point = random_point_in_bbox(min_x, min_y, max_x, max_y, random)
			if contains_any(poly, point):
				return point


def random_points_in_poly(
	poly: shapely.Polygon | shapely.MultiPolygon | geopandas.GeoSeries | geopandas.GeoDataFrame,
	n: int,
	random: RandomSeed = None,
	*,
	use_tqdm: bool = False,
	**tqdm_kwargs,
) -> list[shapely.Point]:
	"""
	Uniformly-ish generates several points somewhere within a polygon.
	This won't choose anywhere directly on the edge (I think). If poly is a MultiPolygon, it will be inside one of the components, but the distribution of which one might not necesarily be uniform.

	Arguments:
		poly: shapely Polygon or MultiPolygon
		random: Optionally a numpy random generator or seed, otherwise default_rng is used
	"""
	if isinstance(poly, (geopandas.GeoDataFrame, geopandas.GeoSeries)):
		min_x, min_y, max_x, max_y = poly.total_bounds
	else:
		min_x, min_y, max_x, max_y = poly.bounds
		shapely.prepare(poly)
	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)
	t = tqdm(**tqdm_kwargs, total=n) if use_tqdm else nullcontext()
	
	out: list[shapely.Point] = []
	with t:
		while len(out) < n:
			points = random_points_in_bbox(n - len(out), min_x, min_y, max_x, max_y, random)
			contains = contains_any_array(poly, points)
			contained_points = points[contains].tolist()
			contained_points = contained_points[:n - len(out)]
			if isinstance(t, tqdm):
				t.update(len(contained_points))
			out += contained_points
	return out
