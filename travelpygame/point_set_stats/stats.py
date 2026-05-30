from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
import shapely
from geopandas import GeoSeries

from travelpygame.util.distance import cartesian_product_distances, geod_distance
from travelpygame.util.geo_utils import (
	circular_mean_xy,
	fix_x_coord,
	fix_y_coord,
	get_area,
	get_geometry_antipode,
)

from .optimize_funcs import find_furthest_point, find_geometric_median

if TYPE_CHECKING:
	from travelpygame.point_set import PointSet


@dataclass
class PointSetStats:
	# centres
	circular_mean: shapely.Point
	arithmetic_mean: shapely.Point
	"""Just the mean of all the lat/lng coordinates"""
	arithmetic_median: shapely.Point
	closest_to_bbox: shapely.Point
	raw_centroid: shapely.Point
	"""Computed using normal geometric CRS, so technically wrong and assumes flat earth"""
	centroid: shapely.Point | None
	"""Optional, computed using projected CRS"""
	centre_of_extremes: shapely.Point
	"""Centre of bounding box"""
	geometric_median: shapely.Point | None
	"""Optional since it can take some time to compute"""
	antipoint: shapely.Point | None
	"""Furthest possible point from anywhere on earth, optional since it can take some time to compute"""

	# extreme points
	# Maybe some of these shouldn't be tuples and should instead be separated into two attributes
	westmost: tuple[float, list[Hashable]]
	"""(longitude, list of indexes at this longitude). Assumes the earth is flat and that -180 longitude is the edge of the planet, because WGS84"""
	eastmost: tuple[float, list[Hashable]]
	"""(longitude, list of indexes at this longitude). Assumes the earth is flat and that 180 longitude is the edge of the planet, because WGS84"""
	northmost: tuple[float, list[Hashable]]
	"""(latitude, list of indexes at this latitude)"""
	southmost: tuple[float, list[Hashable]]
	"""(latitude, list of indexes at this latitude)"""
	nw_most: tuple[float, Hashable]
	"""(distance, index closest to corner)"""
	ne_most: tuple[float, Hashable]
	"""(distance, index closest to corner)"""
	sw_most: tuple[float, Hashable]
	"""(distance, index closest to corner)"""
	se_most: tuple[float, Hashable]
	"""(distance, index closest to corner)"""
	antipoint_closest: Hashable | None

	# Distances in metres that might be a good measure of the extent of one's travels
	diagonal_dist: float
	total_dist_from_centroid: float
	max_dist_from_centroid: float
	antipoint_dist: float | None
	"""Distance from any point in the point set to the antipoint, so smaller numbers indicate more well-travelled (can cover own deadzones better). Optional since the antipoint can take some time to compute"""
	# Other measures of extent
	convex_hull_area: float
	concave_hull_area: float
	bbox_area: float
	num_graticles: int

	# Other stuff
	closest_to_bbox_label: Hashable
	closest_to_bbox_dist: float

	@property
	def centres(self) -> dict[str, shapely.Point]:
		d = {
			'Circular mean point': self.circular_mean,
			'Mean point': self.arithmetic_mean,
			'Median point': self.arithmetic_median,
			'Closest point to bounding box corners': self.closest_to_bbox,
			'Centroid': self.raw_centroid,
			'Centre of extremes': self.centre_of_extremes,
		}
		if self.centroid is not None:
			d['Centroid (projected)'] = self.centroid
		if self.geometric_median is not None:
			d['Geometric median'] = self.geometric_median
		return d

	@property
	def distance_extents(self) -> dict[str, float]:
		return {
			'Diagonal distance of bounding box': self.diagonal_dist,
			'Total distance from centroid': self.total_dist_from_centroid,
			'Maximum distance from centroid': self.max_dist_from_centroid,
		}

	@property
	def area_extents(self) -> dict[str, float]:
		return {
			'Bounding box area': self.bbox_area,
			'Convex hull area': self.convex_hull_area,
			'Concave hull area': self.concave_hull_area,
		}


def get_point_set_stats(
	point_set: 'PointSet',
	*,
	find_geomedian: bool = False,
	find_antipoint: bool = False,
	get_projected_centroid: bool = True,
):
	geo = point_set.points
	coords = shapely.get_coordinates(geo)
	west, south, east, north = geo.total_bounds
	sw = shapely.Point(west, south)
	se = shapely.Point(east, south)
	nw = shapely.Point(west, north)
	ne = shapely.Point(east, north)
	bbox = shapely.box(west, south, east, north)
	x, y = coords.T
	d = dict(point_set.items())

	westmost = geo[geo.x == west].index.tolist()
	eastmost = geo[geo.x == east].index.tolist()
	northmost = geo[geo.y == north].index.tolist()
	southmost = geo[geo.y == south].index.tolist()

	centre_x = fix_x_coord((west + east) / 2)
	centre_y = fix_y_coord((south + north) / 2)
	centre_of_extremes = shapely.Point(centre_x, centre_y)
	circ_mean_x, circ_mean_y = circular_mean_xy(x, y)
	circ_mean = shapely.Point(circ_mean_x, circ_mean_y)
	mean = shapely.Point(fix_x_coord(x.mean()), fix_y_coord(y.mean()))
	median_coords = numpy.median(coords, axis=0)
	median = shapely.Point(fix_x_coord(median_coords[0]), fix_y_coord(median_coords[1]))

	bbox_dists = cartesian_product_distances(
		geo, GeoSeries([sw, se, nw, ne], index=['sw', 'se', 'nw', 'ne'])
	)
	total_bbox_dists = bbox_dists.sum(axis='columns')
	closest_index_to_corners = total_bbox_dists.idxmax()
	closest_to_bbox_dist = total_bbox_dists.loc[closest_index_to_corners]
	closest_to_corners = d[closest_index_to_corners]
	nwmost, nw_dist = point_set.get_closest_index(nw)
	nemost, ne_dist = point_set.get_closest_index(ne)
	swmost, sw_dist = point_set.get_closest_index(sw)
	semost, se_dist = point_set.get_closest_index(se)
	diagonal_dist = geod_distance(sw, ne)

	# This is a really cheesy way of doing this, but I haven't thought of any reason why not to do it
	graticules = point_set.coord_array.astype(int)
	num_unique_graticules = numpy.unique(graticules, sorted=False).shape[0]

	raw_centroid = shapely.centroid(point_set.multipoint)
	if get_projected_centroid:
		centroid = point_set.centroid
		centroid_distances = point_set.get_all_distances(centroid)
	else:
		centroid = None
		centroid_distances = point_set.get_all_distances(raw_centroid)
	total_centroid_dist = centroid_distances.sum()
	max_centroid_dist = centroid_distances.max()

	if find_antipoint:
		initial = get_geometry_antipode(circ_mean)
		antipoint, antipoint_dist = find_furthest_point(geo, initial)
		antipoint_closest, _ = point_set.get_closest_index(antipoint)
	else:
		antipoint = antipoint_dist = antipoint_closest = None
	geo_median = find_geometric_median(geo, centroid) if find_geomedian else None

	return PointSetStats(
		circ_mean,
		mean,
		median,
		closest_to_corners,
		raw_centroid,
		centroid,
		centre_of_extremes,
		geo_median,
		antipoint,
		(west, westmost),
		(east, eastmost),
		(north, northmost),
		(south, southmost),
		(nw_dist, nwmost),
		(ne_dist, nemost),
		(sw_dist, swmost),
		(se_dist, semost),
		antipoint_closest,
		diagonal_dist,
		total_centroid_dist,
		max_centroid_dist,
		antipoint_dist,
		get_area(point_set.convex_hull),
		get_area(point_set.concave_hull),
		get_area(bbox),
		num_unique_graticules,
		closest_index_to_corners,
		closest_to_bbox_dist,
	)
