import logging
from collections import Counter
from collections.abc import Hashable
from functools import cached_property
from typing import Any

import numpy
import pyproj
import shapely
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from shapely.ops import transform

from travelpygame.util import (
	find_first_geom_index,
	get_distances,
	get_projected_crs,
	get_transform_methods,
)

logger = logging.getLogger(__name__)

_generic_projected_crs = pyproj.CRS(
	'+proj=aeqd +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
)


class PointSet:
	"""Stores a point set and other properties on it to make converting between lots of different types easier. Note: .gdf and .points are intended to be read-only, modifying one might or might not update the other."""

	def __init__(self, gdf: GeoDataFrame, name: str, projected_crs: Any | None = None):
		self.gdf: GeoDataFrame = gdf
		self.name = name
		self.points = gdf.geometry
		self.projected_crs_arg = projected_crs
		"""Argument which may be a CRS or a string etc and has not been validated yet"""

	@property
	def count(self) -> int:
		return self.gdf.index.size

	@cached_property
	def point_array(self):
		return self.points.to_numpy()

	@cached_property
	def multipoint(self):
		return shapely.MultiPoint(self.point_array)

	@cached_property
	def coord_array(self):
		return shapely.get_coordinates(self.points)

	@cached_property
	def convex_hull(self):
		return shapely.convex_hull(self.multipoint)

	@cached_property
	def concave_hull(self):
		return shapely.concave_hull(self.multipoint)

	@cached_property
	def envelope(self):
		return shapely.envelope(self.multipoint)

	def contains(self, point: shapely.Point, tolerance: float | None = 1e-6):
		return find_first_geom_index(self.points, point, tolerance) is not None

	def get_all_distances(
		self, target: shapely.Point | tuple[float, float], *, use_haversine: bool = False
	):
		"""Gets distances in metres from all points in this point set to a given target, sorted by closest first."""
		distances = get_distances(target, self.points, use_haversine=use_haversine)
		return Series(distances, index=self.points.index).sort_values()

	@cached_property
	def projected_crs(self):
		if self.projected_crs_arg is None:
			crs = get_projected_crs(self.points)
			if crs:
				return crs
			logger.info('Could not detect projected CRS for %s, using a generic one', self.name)
		return _generic_projected_crs

	@cached_property
	def projected_multipoint(self) -> shapely.MultiPoint:
		"""Returns points projected to a projected CRS, as a MultiPoint."""
		return shapely.MultiPoint(self.points.to_crs(self.projected_crs).to_numpy())

	@cached_property
	def centroid(self):
		"""Takes into account `projected_crs`."""
		proj_centroid = self.projected_multipoint.centroid
		# I guess we could cache from_proj here, but it's not that important
		_to_proj, from_proj = get_transform_methods(self.gdf.crs or 'WGS84', self.projected_crs)
		return transform(from_proj, proj_centroid)


def validate_points(
	geo: GeoSeries | GeoDataFrame,
	rounding_tolerance: int | None = 6,
	name_for_log: Any = None,
	*,
	log_duplicates: bool = True,
) -> tuple[GeoSeries, set[Hashable]]:
	"""Validates a point set to check it has no NaN/infinity coordinates, and removes duplicates (by default, considering points the same if their coordinates rounded down to 6 decimal places are the same). Does not necessarily validate that anything is not a point (at least not properly), you should do that yourself.

	Arguments:
		geo: GeoSeries or GeoDataFrame containing points.
		rounding_tolerance: Number of decimal places to consider when checking for duplicate points, or None to not round any coordinates. This does not modify the point set with the rounded coordinates, it is only for checking.
		name_for_log: Optionally use a specific name for this point set when logging, if anything needs to be logged.

	Returns:
		tuple:
			[0] If any points were invalid or duplicates, the new GeoSeries with the invalid/duplicate points removed (of course keeping the original instance of sets of duplicates), or the original GeoSeries if nothing needed to be dropped.
			[1] Set containing index labels of any points that are removed in the returned GeoSeries.
	"""
	if isinstance(geo, GeoDataFrame):
		geo = geo.geometry

	coords = shapely.get_coordinates(geo)
	if rounding_tolerance is not None:
		coords = coords.round(rounding_tolerance)
	first_points: dict[tuple[float, float], tuple[Hashable, shapely.Point]] = {}
	"""The first instance of every point. Doesn't really need to be a dict I guess."""
	to_drop: set[Hashable] = set()

	for i, (index, item) in enumerate(geo.items()):
		x, y = coords[i]

		checks = (
			(numpy.isnan(x), 'NaN longitude'),
			(numpy.isnan(y), 'NaN latitude'),
			(numpy.isinf(x), 'infinity longitude'),
			(numpy.isinf(y), 'infinity latitude'),
			(x > 180, 'longitude too east'),
			(x < -180, 'longitude too west'),
			(y > 90, 'latitude too north'),
			(y < -90, 'latitude too south'),
		)

		coords_valid = True
		for check, desc in checks:
			if check:
				logger.info('%s had point %s with %s', name_for_log, index, desc)
				coords_valid = False
				break

		if coords_valid:
			if (x, y) in first_points:
				if log_duplicates:
					logger.info(
						'%s had duplicate point %s (identical to %s, %s)',
						name_for_log,
						index,
						*first_points[x, y],
					)
				to_drop.add(index)
			else:
				assert isinstance(item, shapely.Point), (
					f'uh oh item at {index} is actually {type(item)}'
				)
				first_points[x, y] = index, item
		else:
			to_drop.add(index)
	return geo.drop(list(to_drop)) if to_drop else geo, to_drop


def get_visited_regions(point_set: PointSet, regions: GeoDataFrame | GeoSeries):
	"""Finds regions that a point set contains, and that it does not contain, and how often it contains each region.

	Assumes point_set and regions are in the same CRS.

	Returns:
		Counter with keys = indexes in regions and values = amount that this region was visited in the point set, including 0 where it was not visited
	"""
	geo = regions.geometry if isinstance(regions, GeoDataFrame) else regions
	region_indices, _point_indices = point_set.gdf.sindex.query(
		geo, 'contains', output_format='indices'
	)
	indexes = regions.index[region_indices]
	# hrm would it be faster to construct the Counter directly
	counts = indexes.value_counts(sort=True).reindex(regions.index, fill_value=0)
	return Counter(counts.to_dict())
