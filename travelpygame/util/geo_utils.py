"""Utility functions for dealing with geometries that exist on this world, where -180 > x > 180, and -90 > y > 90."""

from collections.abc import Iterable, Sequence
from typing import overload

import numpy
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .distance import wgs84_geod


@overload
def wgs84_to_cartesian(lat: float, lng: float) -> tuple[float, float, float]: ...
@overload
def wgs84_to_cartesian(
	lat: numpy.ndarray, lng: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ...


def wgs84_to_cartesian(
	lat: float | numpy.ndarray, lng: float | numpy.ndarray
) -> tuple[float, float, float] | tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
	phi = numpy.radians(lat)
	lamb = numpy.radians(lng)
	x = numpy.cos(phi) * numpy.cos(lamb)
	y = numpy.cos(phi) * numpy.sin(lamb)
	z = numpy.sin(phi)
	return x, y, z


def get_area(geom: BaseGeometry | GeoSeries | GeoDataFrame) -> float:
	"""Quick shortcut for Geod.get_area_perimeter (that is more convenient to use), but also gets the total area of a GeoSeries/GeoDataFrame if you want to do that."""
	if isinstance(geom, GeoDataFrame):
		return get_area(geom.geometry)
	if isinstance(geom, (GeoSeries, GeometryArray)):
		return sum(geom.map(get_area, na_action='ignore'))
	area, _perimeter = wgs84_geod.geometry_area_perimeter(geom)
	return abs(area)


def get_midpoint(point_a: shapely.Point, point_b: shapely.Point):
	"""Gets the midpoint of 2 points in WGS84 by calculating the direction and distance from one to the other, and then going forward in that direction half that distance."""
	# TODO: Vectorized version
	forward_azimuth, _, dist = wgs84_geod.inv(
		point_a.x, point_a.y, point_b.x, point_b.y, return_back_azimuth=False
	)
	lng, lat, _ = wgs84_geod.fwd(point_a.x, point_a.y, forward_azimuth, dist / 2)
	return shapely.Point(lng, lat)


def get_midpoint_centre(
	point_a: shapely.Point | tuple[float, float], point_b: shapely.Point | tuple[float, float]
):
	"""Gets the midpoint of 2 points by calculating their centre of gravity (as described here: https://geomidpoint.com/calculation.html).

	Assumes both point_a and point_b are in WGS84. Also assumes the earth is a sphere, which it isn't.
	"""
	if isinstance(point_a, shapely.Point):
		lat_a = point_a.y
		lng_a = point_a.x
	else:
		lat_a, lng_a = point_a
	if isinstance(point_b, shapely.Point):
		lat_b = point_b.y
		lng_b = point_b.x
	else:
		lat_b, lng_b = point_b
	x1, y1, z1 = wgs84_to_cartesian(lat_a, lng_a)
	x2, y2, z2 = wgs84_to_cartesian(lat_b, lng_b)

	x = (x1 + x2) / 2
	y = (y1 + y2) / 2
	z = (z1 + z2) / 2

	lamb = numpy.atan2(y, x)
	hyp = numpy.sqrt(x * x + y * y)
	phi = numpy.atan2(z, hyp)

	lat = numpy.degrees(phi)
	lng = numpy.degrees(lamb)

	return shapely.Point(lng, lat)


def get_antipode(lat: float, lng: float) -> tuple[float, float]:
	antilat = -lat
	antilng = lng + 180
	if antilng > 180:
		antilng -= 360
	return antilat, antilng


def get_geometry_antipode[T: BaseGeometry](g: T) -> T:
	if isinstance(g, shapely.Point):
		antilat, antilng = get_antipode(g.y, g.x)
		return type(g)(antilng, antilat)
	return transform(get_antipodes, g)  # ty: ignore[invalid-argument-type] #The type hint is wrong, transform works with arrays just fine


def get_antipodes(lats: 'numpy.ndarray', lngs: 'numpy.ndarray'):
	"""Vectorized version of get_antipode"""
	antilat = -lats
	antilng = lngs + 180
	antilng[antilng > 180] -= 360
	return antilat, antilng


def get_point_antipodes(points: Iterable[shapely.Point] | GeoSeries):
	"""Vectorized version of get_geometry_antipodes"""
	if isinstance(points, (numpy.ndarray, list, tuple, GeoSeries)):
		lngs, lats = shapely.get_coordinates(points).T  # ty:ignore[invalid-argument-type] #not narrowing properly

	else:
		lats_tuple, lngs_tuple = zip(((point.y, point.x) for point in points), strict=True)
		lats = numpy.asarray(lats_tuple)
		lngs = numpy.asarray(lngs_tuple)
	antilats, antilngs = get_antipodes(lats, lngs)
	antipoints = shapely.points(antilngs, antilats)
	assert isinstance(antipoints, numpy.ndarray), f'antipoints is {type(antipoints)}'
	return antipoints


def fix_x_coord(x: float) -> float:
	x %= 360
	if x > 180:
		return x - 360
	if x <= -180:
		return x + 360
	return x


def fix_y_coord(y: float) -> float:
	y %= 180
	if y > 90:
		return y - 180
	if y <= -90:
		return y + 180
	return y


def circular_mean(angles: Sequence[float] | numpy.ndarray) -> float:
	"""Assumes angles are in radians.

	Returns:
		Mean angle"""
	if not isinstance(angles, numpy.ndarray):
		angles = numpy.asarray(angles)
	sin_sum = numpy.sin(angles).sum()
	cos_sum = numpy.cos(angles).sum()
	# Convert it from numpy.floating to float otherwise that's maybe annoying
	return numpy.atan2(sin_sum, cos_sum).item()


def circular_mean_xy(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
	"""x and y are assumed to be convertible to numpy.ndarray! I can't be arsed type hinting list-like

	Returns:
		mean of x, mean of y
		i.e. long and then lat, do not get them swapped around I swear on me mum
	"""
	if not isinstance(x, (numpy.ndarray)):
		x = numpy.asarray(x)
	if not isinstance(y, (numpy.ndarray)):
		y = numpy.asarray(y)
	x = numpy.radians(x + 180)  # ty:ignore[unsupported-operator] #hrm it seems to think x can contain tuples
	y = numpy.radians((y + 90) * 2)  # ty:ignore[unsupported-operator] #same here with y
	mean_x = numpy.degrees(circular_mean(x))
	mean_y = numpy.degrees(circular_mean(y))
	mean_x = (mean_x % 360) - 180
	mean_y = ((mean_y % 360) / 2) - 90
	return mean_x, mean_y


def circular_mean_points(
	points: Sequence[BaseGeometry] | GeoSeries | numpy.ndarray,
) -> shapely.Point:
	"""Returns the point with coordinates being the circular mean of all coordinates in points."""
	coords = shapely.get_coordinates(points)
	x, y = coords.T
	mean_x, mean_y = circular_mean_xy(x, y)
	return shapely.Point(mean_x, mean_y)


def geod_buffer_as_line(
	point: shapely.Point | tuple[float, float], distance: float, segments: int = 32
) -> shapely.LineString:
	"""This will fail miserably for distances being too high and wrapping around the world… hrm."""
	if isinstance(point, shapely.Point):
		lat = point.y
		lng = point.x
	else:
		lat, lng = point
	azimuths = numpy.linspace(0, 360, segments)

	lngs, lats, _ = wgs84_geod.fwd(
		numpy.repeat(lng, segments),
		numpy.repeat(lat, segments),
		azimuths,
		numpy.repeat(distance, segments),
	)
	return shapely.LineString(numpy.column_stack((lngs, lats)))


def _geod_buffer_as_arc_poly(lat: float, lng: float, distance: float, quad_segs: int, quad: int):
	"""Important to do things one quadrant at a time to avoid meridian nonsense, or does it"""
	start = 90 * quad
	end = 90 * (quad + 1)
	azimuths = numpy.linspace(start, end, quad_segs)

	lngs, lats, _ = wgs84_geod.fwd(
		numpy.repeat(lng, repeats=quad_segs),
		numpy.repeat(lat, quad_segs),
		azimuths,
		numpy.repeat(distance, quad_segs),
	)
	lngs = numpy.append(lngs, lng)
	lats = numpy.append(lats, lat)
	ring = shapely.LinearRing(numpy.column_stack([lngs, lats]))
	return shapely.Polygon(ring)


def geod_buffer(point: shapely.Point | tuple[float, float], distance: float, quad_segs: int = 8):
	"""Like shapely.buffer but for geodetic distances, but it currently behaves strangely with large distances anyway. Not quite an issue of antimeridian wrapping, just ends up in the wrong place sometimes…"""
	if isinstance(point, shapely.Point):
		lat = point.y
		lng = point.x
	else:
		lat, lng = point

	quadrants = [_geod_buffer_as_arc_poly(lat, lng, distance, quad_segs, i) for i in range(4)]
	return shapely.MultiPolygon(quadrants)


def geod_buffer_line(line: shapely.LineString, distance: float, quad_segs: int = 8):
	# TODO: Also need to buffer segments in between coords if distance > space between coords, or else this won't work and will return multipolygon (which will fail the assert) so this wouldn't work for now
	buffers = [geod_buffer((x, y), distance, quad_segs) for x, y in line.coords]
	joined = shapely.union_all(buffers)
	assert isinstance(joined, shapely.Polygon)
	return joined


def mean_points(points: Sequence[BaseGeometry] | GeoSeries | numpy.ndarray) -> shapely.Point:
	"""Returns a point with the lat being the mean of each point's latitude, and longitude being the mean of each point's longitude, etc. without any circular business."""
	coords = shapely.get_coordinates(points)
	# This could even support include_z and include_m if we really wanted
	mean = coords.mean(axis=0)
	mean[0] = fix_x_coord(mean[0])
	mean[1] = fix_y_coord(mean[1])
	return shapely.Point(mean)
