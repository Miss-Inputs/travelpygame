"""Tools for measuring distance and such."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numpy
import pandas
import pyproj

if TYPE_CHECKING:
	from shapely import Point

wgs84_geod = pyproj.Geod(ellps='WGS84')

FloatListlike = Sequence[float] | numpy.ndarray | pandas.Series
"""Accepted input types to pyproj.Geod.inv, although other stuff would probably work, this is just what works as a type hint."""


@overload
def geod_distance_and_bearing(
	lat1: float, lng1: float, lat2: float, lng2: float, *, radians: bool = False
) -> tuple[float, float]: ...


@overload
def geod_distance_and_bearing(
	lat1: FloatListlike,
	lng1: FloatListlike,
	lat2: FloatListlike,
	lng2: FloatListlike,
	*,
	radians: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]: ...


def geod_distance_and_bearing(
	lat1: float | FloatListlike,
	lng1: float | FloatListlike,
	lat2: float | FloatListlike,
	lng2: float | FloatListlike,
	*,
	radians: bool = False,
) -> tuple[float | numpy.ndarray, float | numpy.ndarray]:
	"""
	Calculates the WGS84 geodesic distance and heading from one point to another. lat1/lng1/lat2/lng2 can either all be floats, or all arrays.

	Arguments:
		lat1: Latitude of point A, or list/ndarray/etc
		lng1: Longitude of point A, or list/ndarray/etc
		lat2: Latitude of point B, or list/ndarray/etc
		lng2: Longitude of point B, or list/ndarray/etc
		radians: If true, treats the arguments as being in radians, otherwise they are degrees (as normal people use for coordinates)

	Returns:
		(Distance in metres, heading/direction/bearing/whatever you call it from lat1,lng1 to lat2,lng2 in degrees/radians) between point A and point B. If input is an array, it will return an array for each pair of coordinates.
	"""
	bearing, _, dist = wgs84_geod.inv(lng1, lat1, lng2, lat2, radians=radians)
	if isinstance(bearing, list):
		# y u do this
		bearing = numpy.array(bearing)
	return (dist, bearing)


def geod_distance(point1: 'Point', point2: 'Point') -> float:
	"""Returns WGS84 geodesic distance between point1 and point2 (assumed to be WGS84 coordinates) in metres."""
	return geod_distance_and_bearing(point1.y, point1.x, point2.y, point2.x)[0]


@overload
def haversine_distance(
	lat1: float, lng1: float, lat2: float, lng2: float, *, radians: bool = False
) -> float: ...


@overload
def haversine_distance(
	lat1: numpy.ndarray,
	lng1: numpy.ndarray,
	lat2: numpy.ndarray,
	lng2: numpy.ndarray,
	*,
	radians: bool = False,
) -> numpy.ndarray: ...


def haversine_distance(
	lat1: float | numpy.ndarray,
	lng1: float | numpy.ndarray,
	lat2: float | numpy.ndarray,
	lng2: float | numpy.ndarray,
	*,
	radians: bool = False,
) -> float | numpy.ndarray:
	"""Calculates haversine distance (which TPG uses), treating the earth as a sphere.

	Arguments:
		lat1: ndarray of floats
		lng1: ndarray of floats
		lat1: ndarray of floats
		lng1: ndarray of floats
		radians: If set to true, treats the lat/long arguments as being in radians, otherwise they are treated as degrees (as normal people would use for coordinates)

	Returns:
		ndarray (float) of distances in metres

	"""
	r = 6371_000
	if not radians:
		lat1 = numpy.radians(lat1)
		lat2 = numpy.radians(lat2)
		lng1 = numpy.radians(lng1)
		lng2 = numpy.radians(lng2)
	dlng = lng2 - lng1
	dlat = lat2 - lat1
	a = (numpy.sin(dlat / 2) ** 2) + numpy.cos(lat1) * numpy.cos(lat2) * (numpy.sin(dlng / 2) ** 2)
	c = 2 * numpy.asin(numpy.sqrt(a))
	if isinstance(c, numpy.floating):
		# Just to make sure nothing annoying happens elsewhere
		c = c.item()
	return c * r
