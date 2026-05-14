"""Utilities to work with projected CRSes and do operations with geographical coordinates that would otherwise expect projected coordinates."""

from collections.abc import Callable
from functools import cache, partial
from typing import Any

import pyproj
import shapely
from geopandas import GeoDataFrame, GeoSeries
from pyproj.aoi import AreaOfInterest
from pyproj.database import CRSInfo, query_crs_info
from pyproj.enums import PJType
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


def get_projected_crs(
	bounds: tuple[float, float, float, float] | BaseGeometry | GeoSeries | GeoDataFrame,
) -> pyproj.CRS | None:
	"""Similar to geopandas estimate_utm_crs, but doesn't necessarily use an UTM zone. Tries to manually exclude projected CRSes that aren't suitable for these use cases. Might be overcomplicated."""
	if isinstance(bounds, (GeoSeries, GeoDataFrame)):
		west, south, east, north = bounds.total_bounds
	elif isinstance(bounds, BaseGeometry):
		west, south, east, north = bounds.bounds
	else:
		west, south, east, north = bounds

	aoi = AreaOfInterest(west, south, east, north)
	crs_infos = query_crs_info(area_of_interest=aoi, pj_types=PJType.PROJECTED_CRS, contains=True)
	crs_infos = [
		info
		for info in crs_infos
		if info.auth_name != 'IAU_2015' and info.name != 'GOES-16_East_ABI_Fixed_Grid_ITRF2008'
	]
	if not crs_infos:
		return None
	# Most certainly a better way to do that but eh
	crs_infos = sorted(
		crs_infos,
		key=lambda info: (
			not info.name.startswith('GDA2020 /'),
			info.projection_method_name != 'Albers Equal Area',
			info.code,
		),
	)
	sort_key = _make_crs_info_sort_key((west, south, east, north))
	crs_infos = sorted(crs_infos, key=sort_key)
	valid_ellipses = pyproj.get_ellps_map().keys() | {'GRS 1980'}
	# This seems to have all the _earth_ ellipsoids
	for crs_info in crs_infos:
		crs = pyproj.CRS.from_authority(crs_info.auth_name, crs_info.code)
		if crs.datum and crs.datum.ellipsoid and crs.datum.ellipsoid.name in valid_ellipses:
			return crs
	return None


def bounds_distance(
	bounds: tuple[float, float, float, float], other_bounds: tuple[float, float, float, float]
):
	"""Total distance in degrees between two bounding boxes, used for sorting/comparison."""
	return sum(abs(bounds[i] - other_bounds[i]) % 360 for i in range(4))


def _make_crs_info_sort_key(bounds: tuple[float, float, float, float]):
	def sort_key(crs_info: CRSInfo):
		aou = crs_info.area_of_use
		if aou is None:
			return float('inf')
		return bounds_distance(bounds, (aou.west, aou.south, aou.east, aou.north))

	return sort_key


def get_metric_crs(g: BaseGeometry, *, query_db: bool = True) -> pyproj.CRS:
	"""Returns a CRS that uses metres as its unit and that can be used with a particular geometry."""
	if isinstance(g, shapely.Point):
		point = g
	else:
		if query_db:
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
