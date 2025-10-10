from collections.abc import Callable, Iterable, Sequence
from functools import cache, partial
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy
import pyproj
import shapely
from geopandas import GeoDataFrame, GeoSeries
from pyproj.aoi import AreaOfInterest
from pyproj.database import CRSInfo, query_crs_info
from pyproj.enums import PJType
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import transform

from .distance import wgs84_geod

if TYPE_CHECKING:
	from geopandas.array import GeometryArray


def get_poly_vertices(poly: shapely.Polygon | shapely.MultiPolygon) -> list[shapely.Point]:
	if isinstance(poly, shapely.MultiPolygon):
		return list(chain.from_iterable(get_poly_vertices(part) for part in poly.geoms))
	out = shapely.points(poly.exterior.coords)
	if isinstance(out, shapely.Point):
		return [out]
	return out.tolist()


def get_midpoint(point_a: shapely.Point, point_b: shapely.Point):
	# TODO: Alternate calculation that matches what other tools of this nature do
	# TODO: Vectorized version
	forward_azimuth, _, dist = wgs84_geod.inv(
		point_a.x, point_a.y, point_b.x, point_b.y, return_back_azimuth=False
	)
	lng, lat, _ = wgs84_geod.fwd(point_a.x, point_a.y, forward_azimuth, dist / 2)
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
		return shapely.Point(antilng, antilat)
	return transform(get_antipodes, g)  # pyright: ignore[reportArgumentType] #The type hint is wrong, transform works with arrays just fine


def get_antipodes(lats: 'numpy.ndarray', lngs: 'numpy.ndarray'):
	"""Vectorized version of get_antipode"""
	antilat = -lats
	antilng = lngs + 180
	antilng[antilng > 180] -= 360
	return antilat, antilng


def get_point_antipodes(points: Iterable[shapely.Point] | GeoSeries):
	"""Vectorized version of get_geometry_antipodes"""
	if isinstance(points, GeoSeries):
		lats = points.y.to_numpy()
		lngs = points.x.to_numpy()
	elif isinstance(points, (numpy.ndarray, list, tuple)):
		lngs, lats = shapely.get_coordinates(points).T
	else:
		lats_tuple, lngs_tuple = zip(((point.y, point.x) for point in points), strict=True)
		lats = numpy.asarray(lats_tuple)
		lngs = numpy.asarray(lngs_tuple)
	antilats, antilngs = get_antipodes(lats, lngs)
	antipoints = shapely.points(antilngs, antilats)
	assert isinstance(antipoints, numpy.ndarray), f'antipoints is {type(antipoints)}'
	return antipoints


def bounds_distance(
	bounds: tuple[float, float, float, float], other_bounds: tuple[float, float, float, float]
):
	return sum(abs(bounds[i] - other_bounds[i]) % 360 for i in range(4))


def _make_crs_info_sort_key(bounds: tuple[float, float, float, float]):
	def sort_key(crs_info: CRSInfo):
		aou = crs_info.area_of_use
		if aou is None:
			return float('inf')
		return bounds_distance(bounds, (aou.west, aou.south, aou.east, aou.north))

	return sort_key


def get_projected_crs(
	bounds: tuple[float, float, float, float] | BaseGeometry | GeoSeries | GeoDataFrame,
) -> pyproj.CRS | None:
	"""Similar to geopandas estimate_utm_crs, but doesn't necessarily use an UTM zone"""
	if isinstance(bounds, (GeoSeries, GeoDataFrame)):
		west, south, east, north = bounds.total_bounds
	elif isinstance(bounds, BaseGeometry):
		west, south, east, north = bounds.bounds
	else:
		west, south, east, north = bounds
	aoi = AreaOfInterest(west, south, east, north)
	crs_infos = query_crs_info(area_of_interest=aoi, pj_types=PJType.PROJECTED_CRS, contains=True)
	crs_infos = [info for info in crs_infos if info.auth_name != 'IAU_2015']
	if not crs_infos:
		return None
	# Most certainly a better way to do that but eh
	crs_infos = sorted(
		crs_infos, key=lambda info: (info.code, info.name.startswith('GDA2020')), reverse=True
	)
	sort_key = _make_crs_info_sort_key((west, south, east, north))
	crs_infos = sorted(crs_infos, key=sort_key)
	valid_ellipses = pyproj.get_ellps_map().keys()
	# This seems to have all the _earth_ ellipsoids
	for crs_info in crs_infos:
		crs = pyproj.CRS.from_authority(crs_info.auth_name, crs_info.code)
		if crs.datum and crs.datum.ellipsoid and crs.datum.ellipsoid.name in valid_ellipses:
			return crs
	return None


def get_metric_crs(g: BaseGeometry) -> pyproj.CRS:
	"""Returns a CRS that uses metres as its unit and that can be used with a particular geometry."""
	if isinstance(g, shapely.Point):
		point = g
	else:
		projected = get_projected_crs(g)
		if projected:
			return projected
		point = g.representative_point()
	return pyproj.CRS(
		f'+proj=aeqd +lat_0={point.y} +lon_0={point.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
	)


@cache
def get_transform_methods(from_crs: Any, to_crs: Any):
	"""Gets two functions to use with shapely.ops.transform, the regular direction, and inverse. Less verbose this way."""
	transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
	return transformer.transform, partial(transformer.transform, direction='inverse')


def apply_transformed[T: 'BaseGeometry'](
	func: Callable[..., T],
	g: 'BaseGeometry',
	crs: Any = None,
	from_crs: Any = 'WGS84',
	*args,
	**kwargs,
) -> T:
	if not crs:
		crs = get_metric_crs(g)
	transform_to, transform_from = get_transform_methods(from_crs, crs)
	projected = transform(transform_to, g)
	result = func(projected, *args, **kwargs)
	return transform(transform_from, result)


def get_centroid(g: 'BaseGeometry', crs: Any = None, from_crs: Any = 'WGS84') -> shapely.Point:
	"""Gets the centroid of some points in WGS84 properly, accounting for projection by converting to a different CRS instead."""
	return apply_transformed(shapely.centroid, g, crs, from_crs)


def fix_x_coord(x: float) -> float:
	x %= 360
	if x > 180:
		return x - 360
	if x <= 180:
		return x + 360
	return x


def fix_y_coord(y: float) -> float:
	y %= 180
	if y > 90:
		return y - 180
	if y <= 90:
		return y + 180
	return y


def circular_mean(angles: Sequence[float] | numpy.ndarray) -> float:
	"""Assumes this is in radians

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
	x = numpy.radians(x + 180)
	y = numpy.radians((y + 90) * 2)
	mean_x = numpy.degrees(circular_mean(x))
	mean_y = numpy.degrees(circular_mean(y))
	mean_x = (mean_x % 360) - 180
	mean_y = ((mean_y % 360) / 2) - 90
	return mean_x, mean_y


def circular_mean_points(points: Iterable[shapely.Point]) -> shapely.Point:
	"""points is assumed to be convertible to numpy.ndarray!"""
	x, y = zip(*((a.x, a.y) for a in points), strict=True)
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
		numpy.repeat(lng, quad_segs),
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


def contains_any(
	geo: 'GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry',
	point: shapely.Point | tuple[float, float],
) -> bool:
	"""Returns a bool indicating if a point is anywhere within any part of a geometry or GeoPandas object."""
	if isinstance(geo, BaseGeometry):
		return (
			shapely.contains_xy(geo, point[1], point[0]).item()
			if isinstance(point, tuple)
			else geo.contains(point)
		)
	if isinstance(point, tuple):
		point = shapely.Point(point)
	# Unfortunately there is no contains_properly inverse
	return geo.sindex.query(point, 'within').size > 0


def contains_any_array(
	geo: 'GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry',
	points: numpy.ndarray | Sequence[shapely.Point],
):
	"""Returns an array of of booleans for each point, indicating whether each point is anywhere in a geometry or GeoPandas object."""
	if isinstance(geo, BaseGeometry):
		return geo.contains(points)
	# Unfortunately there is no contains_properly inverse
	if not isinstance(points, numpy.ndarray):
		points = numpy.asarray(points)
	return geo.sindex.query(points, 'within', output_format='dense').any(axis=0)


def get_polygons(
	geom: 'GeoDataFrame | GeoSeries | GeometryArray | BaseGeometry',
) -> list[shapely.Polygon]:
	"""Gets every individual polygon inside geom."""
	if isinstance(geom, shapely.Polygon):
		return [geom]
	if isinstance(geom, BaseMultipartGeometry):
		return list(chain.from_iterable(get_polygons(part) for part in geom.geoms))
	if isinstance(geom, GeoDataFrame):
		geom = geom.geometry
	if not isinstance(geom, BaseGeometry):
		# GeoSeries/array/etc
		return list(chain.from_iterable(get_polygons(item) for item in geom.dropna()))
	# Some other geometry, just silently return nothing
	return []

def find_first_geom_index(gdf: GeoDataFrame|GeoSeries, geom: 'BaseGeometry', tolerance: float |None= 1e-6):
	if isinstance(gdf, GeoDataFrame):
		gdf = gdf.geometry
	if geom in gdf or tolerance is None:
		rows = gdf[gdf == geom]
	else:
		rows = gdf[gdf.geom_equals_exact(geom, tolerance)]
	if rows.empty:
		return None
	return rows.index[0]