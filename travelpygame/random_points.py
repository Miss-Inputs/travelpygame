from contextlib import nullcontext

import numpy
import shapely
from tqdm.auto import tqdm

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


def random_point_in_poly(
	poly: shapely.Polygon | shapely.MultiPolygon,
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
	min_x, max_x, min_y, max_y = poly.bounds
	shapely.prepare(poly)
	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)
	t = tqdm(**tqdm_kwargs) if use_tqdm else nullcontext()
	with t:
		while True:
			if isinstance(t, tqdm):
				t.update(1)
			point = random_point_in_bbox(min_x, max_x, min_y, max_y, random)
			if poly.contains_properly(point):
				return point


def random_points_in_poly(
	poly: shapely.Polygon | shapely.MultiPolygon,
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
	min_x, max_x, min_y, max_y = poly.bounds
	shapely.prepare(poly)
	if not isinstance(random, numpy.random.Generator):
		random = numpy.random.default_rng(random)
	t = tqdm(**tqdm_kwargs, total=n) if use_tqdm else nullcontext()
	points: list[shapely.Point] = []
	with t:
		while len(points) < n:
			point = random_point_in_bbox(min_x, max_x, min_y, max_y, random)
			if poly.contains_properly(point):
				if isinstance(t, tqdm):
					t.update(1)
				points.append(point)
	return points
