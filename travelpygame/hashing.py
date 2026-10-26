"""Hashing functions for coordinates. Used for e.g. the H round in Alphabet TPG"""

import math
import random
from collections.abc import Callable, Iterable, Iterator
from itertools import accumulate, islice, product
from typing import Any

from shapely import Point

from travelpygame.util import DistanceMethod, get_distance

type Hasher = Callable[[float, float], tuple[float, float]]


def _half(x: float, _y: Any) -> float:
	return x / 2.0


def josh_hash(lat: float, lng: float) -> tuple[float, float]:
	"""Recreation of Google Sheets hashing formula used for Hashing TPG round of Alphabet TPG.
	The variable names here are going to suck and the code is going to be a mess. Strap yourself in.
	If you think you can simplify this, good for you, I won't be doing that yet.
	Accurate to the original formula by 8 decimal places, and then differences between Python and Google Sheets start being weird.
	"""
	half_seq = list(accumulate(range(23), _half, initial=0.5))
	"""H1 to AE1: Sequence starting at 0.5 and halving the previous one every time. AI1 to BF1 is identical. We can actually name a variable yay"""

	d2 = math.sin(math.pi * lat / 180)
	e2 = lng / 180

	f2 = -1 if d2 <= 0 else 1
	"""This also has an if statement that returns 1 if lat and lng are empty string, but I don't think that is applicable here"""
	g2 = -d2 if d2 <= 0 else d2
	"""Is that not just abs(d2) or am I cooked/washed"""

	h2_ae2 = [g2 - half_seq[0] if g2 > half_seq[0] else g2] * 24
	for i in range(1, 24):
		# For column in I:AE
		prev = h2_ae2[i - 1]
		h2_ae2[i] = prev - half_seq[i] if prev > half_seq[i] else prev

	_ag2 = -1 if e2 <= 0 else 1
	"""Similar "if empty string" check to f2 in the original sheet. However this was supposed to be referenced in AF4 and wasn't, due to a mistake"""
	ah2 = -e2 if e2 <= 0 else e2
	ai2_bf2 = [ah2 - half_seq[0] if ah2 > half_seq[0] else ah2] * 24
	for i in range(1, 24):
		# For column in AJ:BF
		prev = ai2_bf2[i - 1]
		ai2_bf2[i] = prev - half_seq[i] if prev > half_seq[i] else prev

	# no row 3 or 5, you know, to make the formula human-readable, clearly
	g4 = 0 if f2 < 0 else 1
	h4_ae4 = [((g2 - h2_ae2[0]) / half_seq[0]) % 2] * 24
	# There's for sure a better way to write these loops but I'm not gonna overthink it right now and I just want to make sure it works
	# Like maybe it could be cleaner to just initialize it as [0] * 24, and then set h4_ae4[0] = blah on the next line
	for i in range(1, 24):
		h4_ae4[i] = ((h2_ae2[i - 1] - h2_ae2[i]) / half_seq[i]) % 2

	af4 = 0 if h2_ae2[-2] < 0 else 1
	"""Ruh roh! This wasn't supposed to point at AD2! But we need to preserve the original mistake"""
	ag4_bd4 = [((ah2 - ai2_bf2[0]) / half_seq[0]) % 2] * 24
	for i in range(1, 24):
		# For column in AH:BD
		ag4_bd4[i] = ((ai2_bf2[i - 1] - ai2_bf2[i]) / half_seq[i]) % 2

	g6_ad9 = ((7, 25, 5, 21, 22, 6, 23, 11, 13, 3, 48, 4, 28, 2, 12, 29, 18, 8, 42, 15, 24, 1, 37, 32, 39), (9, 43, 36, 17, 4, 5, 27, 35, 8, 29, 31, 26, 23, 41, 24, 18, 47, 10, 45, 3, 19, 2, 34, 14, 48), (24, 33, 2, 4, 11, 47, 17, 1, 30, 39, 42, 22, 6, 32, 8, 49, 5, 18, 7, 45, 29, 38, 20, 25, 36), (37, 48, 21, 22, 3, 34, 47, 4, 17, 12, 18, 16, 46, 20, 38, 32, 8, 29, 14, 11, 44, 33, 40, 43, 50))  # fmt: skip
	"""Mysterious Josh-generated constants. 4 rows and 25 columns"""
	ah6_bf9 = ((46, 26, 50, 40, 9, 30, 19, 20, 49, 45, 36, 27, 33, 16, 38, 17, 10, 47, 14, 44, 41, 35, 31, 34, 43), (20, 49, 7, 21, 50, 32, 46, 16, 39, 11, 33, 15, 40, 12, 22, 6, 13, 42, 44, 37, 28, 1, 38, 25, 30), (23, 12, 13, 41, 48, 37, 43, 44, 50, 40, 31, 10, 19, 21, 46, 16, 35, 9, 28, 15, 26, 34, 14, 27, 3), (31, 2, 39, 26, 28, 13, 42, 1, 10, 15, 23, 6, 36, 41, 30, 45, 7, 49, 25, 19, 9, 27, 35, 24, 5))  # fmt: skip
	"""Mysterious Josh-generated constants. 4 rows and 25 columns"""
	# need to remember, first index is the row…

	# Original: g10 = =MOD(INDEX($G$4:$BD$4,1,G6)+INDEX($G$4:$BD$4,1,G7)+INDEX($G$4:$BD$4,1,G8)+INDEX($G$4:$BD$4,1,G9),2)*2-1
	# h10 =MOD(INDEX($G$4:$BD$4,1,H6)+INDEX($G$4:$BD$4,1,H7)+INDEX($G$4:$BD$4,1,H8)+INDEX($G$4:$BD$4,1,H9),2)*H1
	# i10 =MOD(INDEX($G$4:$BD$4,1,I6)+INDEX($G$4:$BD$4,1,I7)+INDEX($G$4:$BD$4,1,I8)+INDEX($G$4:$BD$4,1,I9),2)*I1
	# You know what, let's make that reference easier
	# We can even make it use 1-based indexing like INDEX() does
	g4_bd4 = [0, g4, *h4_ae4, af4, *ag4_bd4]
	g10 = (
		(g4_bd4[g6_ad9[0][0]] + g4_bd4[g6_ad9[1][0]] + g4_bd4[g6_ad9[2][0]] + g4_bd4[g6_ad9[3][0]])
		% 2
	) * 2 - 1
	"""That's pointing to each row in G6:G9, I'm definitely smelling a simplification here"""
	h10_ac10 = [
		(
			(
				g4_bd4[g6_ad9[0][i + 1]]
				+ g4_bd4[g6_ad9[1][i + 1]]
				+ g4_bd4[g6_ad9[2][i + 1]]
				+ g4_bd4[g6_ad9[3][i + 1]]
			)
			% 2
		)
		* half_seq[i]
		for i in range(22)
	]
	"""AD10 and AE10 are part of this array but appear to not even be used?"""

	ah10 = (
		(
			g4_bd4[ah6_bf9[0][0]]
			+ g4_bd4[ah6_bf9[1][0]]
			+ g4_bd4[ah6_bf9[2][0]]
			+ g4_bd4[ah6_bf9[3][0]]
		)
		% 2
	) * 2 - 1
	ai10_bd10 = [
		(
			(
				g4_bd4[ah6_bf9[0][i + 1]]
				+ g4_bd4[ah6_bf9[1][i + 1]]
				+ g4_bd4[ah6_bf9[2][i + 1]]
				+ g4_bd4[ah6_bf9[3][i + 1]]
			)
			% 2
		)
		* half_seq[i]
		for i in range(22)
	]
	"""BE10 and BF10 also go unused"""

	d10 = sum(h10_ac10) * g10
	e10 = sum(ai10_bd10) * ah10
	out_lat = math.asin(d10) * 180 / math.pi
	out_lng = e10 * 180

	return (out_lat, out_lng)


def josh_hash_point(point: Point) -> Point:
	lat, lng = josh_hash(point.y, point.x)
	return Point(lng, lat)


def frange(start: float, stop: float, step: float) -> Iterator[float]:
	while start < stop:
		yield start
		start += step


def _compare_coords(lat1: float, lng1: float, lat2: float, lng2: float, precision: int) -> int:
	tol = 10**-precision
	lat_diff = not math.isclose(lat1, lat2, rel_tol=0, abs_tol=tol)
	lng_diff = not math.isclose(lng1, lng2, rel_tol=0, abs_tol=tol)
	return lat_diff + lng_diff


def _reverse_search(
	hasher: Hasher,
	candidates: Iterable[tuple[float, float]],
	target_lat: float,
	target_lng: float,
	threshold: float | None = None,
	distance_method: DistanceMethod | None = None,
	precision: int = 5,
	*,
	early_return: bool = False,
) -> list[tuple[float, float, float, float, float]]:
	# TODO: This should be using itertools.batched or whatever to get vectorized distances, probably
	found = []
	for lat, lng in candidates:
		hashed_lat, hashed_lng = hasher(lat, lng)
		raw_dist = _compare_coords(target_lat, target_lng, hashed_lat, hashed_lng, precision)
		if distance_method is None:
			dist = raw_dist
		else:
			dist = get_distance((target_lat, target_lng), (hashed_lat, hashed_lng), distance_method)

		is_match = dist <= threshold if threshold and threshold > 0 else raw_dist == 0
		if is_match:
			found.append((lat, lng, hashed_lat, hashed_lng, dist))
		if early_return and raw_dist == 0:
			break
	return found


def grid_search(
	target_lat: float,
	target_lng: float,
	threshold: float | None = None,
	distance_method: DistanceMethod | None = None,
	grid_size: float = 1.0,
	precision: int = 5,
	bbox: tuple[float, float, float, float] | None = None,
	*,
	early_return: bool = False,
) -> list[tuple[float, float, float, float, float]]:
	"""Searches possible points for what hashes to the target coordinates.
	Arguments:
		threshold: Allowed distance in metres. If None, only return exact matches up to `precision`.
		distance_method: Method to check that hashed points are within `threshold`. If None, return 0 for exact matches and 1 if only one of lat/lng is correct, otherwise 2.
		grid_size: Size in degrees of the grid.
		precision: Number of decimal places to check, defaults to 5.
		early_return: Exit out of the loop once we have found coordinates that match.
	Returns:
		List of input points and their distance as (lat, lng, hashed lat, hashed lng, distance).
	"""
	# TODO: Argument for hash function (lat,lng -> different lat,lng) to use that defaults to josh_hash, if we ever have more than one
	# TODO: grid_size could optionally be a tuple, for different lat and lng grid size

	if bbox:
		min_lng, min_lat, max_lng, max_lat = bbox
	else:
		min_lng = -180
		min_lat = -90
		max_lng = 180
		max_lat = 90

	lat_gen = frange(min_lat, max_lat, grid_size)
	lng_gen = frange(min_lng, max_lng, grid_size)
	generator = product(lat_gen, lng_gen)

	return _reverse_search(
		josh_hash,
		generator,
		target_lat,
		target_lng,
		threshold,
		distance_method,
		precision,
		early_return=early_return,
	)


def grid_search_point(
	target: Point,
	threshold: float | None = None,
	distance_method: DistanceMethod | None = None,
	grid_size: float = 1.0,
	precision: int = 5,
	bbox: tuple[float, float, float, float] | None = None,
	*,
	early_return: bool = False,
) -> list[tuple[Point, Point, float]]:
	return [
		(Point(lng, lat), Point(hashed_lng, hashed_lat), dist)
		for lat, lng, hashed_lat, hashed_lng, dist in grid_search(
			target.y,
			target.x,
			threshold,
			distance_method,
			grid_size,
			precision,
			bbox,
			early_return=early_return,
		)
	]


def _yield_random_coords(
	bbox: tuple[float, float, float, float] | None,
) -> Iterator[tuple[float, float]]:
	if bbox:
		min_lng, min_lat, max_lng, max_lat = bbox
	else:
		min_lng = -180
		min_lat = -90
		max_lng = 180
		max_lat = 90
	while True:
		lat = random.uniform(min_lat, max_lat)
		lng = random.uniform(min_lng, max_lng)
		yield lat, lng


def random_search(
	target_lat: float,
	target_lng: float,
	threshold: float | None = None,
	distance_method: DistanceMethod | None = None,
	limit: int | None = None,
	precision: int = 5,
	bbox: tuple[float, float, float, float] | None = None,
	*,
	early_return: bool = False,
) -> list[tuple[float, float, float, float, float]]:
	"""Searches possible points for what hashes to the target coordinates.
	Arguments:
		threshold: Allowed distance in metres. If None, only return exact matches up to `precision`.
		distance_method: Method to check that hashed points are within `threshold`. If None, return 0 for exact matches and 1 if only one of lat/lng is correct, otherwise 2.
		limit: Generate up to this many random points before giving up, or if None/0, keep going until a match is found (forces early_return=True).
		precision: Number of decimal places to check, defaults to 5.
		early_return: Exit out of the loop once we have found coordinates that match.
	Returns:
		List of input points and their distance as (lat, lng, hashed lat, hashed lng, distance).
	"""
	# TODO: Argument for random seed I guess
	generator = islice(_yield_random_coords(bbox), limit or None)

	return _reverse_search(
		josh_hash,
		generator,
		target_lat,
		target_lng,
		threshold,
		distance_method,
		precision,
		early_return=early_return or not limit,
	)


def random_search_point(
	target: Point,
	threshold: float | None = None,
	distance_method: DistanceMethod | None = None,
	limit: int | None = None,
	precision: int = 5,
	bbox: tuple[float, float, float, float] | None = None,
	*,
	early_return: bool = False,
) -> list[tuple[Point, Point, float]]:
	return [
		(Point(lng, lat), Point(hashed_lng, hashed_lat), dist)
		for lat, lng, hashed_lat, hashed_lng, dist in random_search(
			target.y,
			target.x,
			threshold,
			distance_method,
			limit,
			precision,
			bbox,
			early_return=early_return,
		)
	]
