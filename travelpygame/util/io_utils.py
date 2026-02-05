"""Stuff for reading/writing files easier, generally pandas or geopandas objects."""

import asyncio
import logging
import sys
import warnings
from collections.abc import Hashable
from pathlib import Path, PurePath
from typing import Any

import geopandas
import pandas
from shapely import Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from tqdm.auto import tqdm

if sys.version_info >= (3, 14):
	from compression import zstd
else:
	from backports import zstd

from .pandas_utils import find_first_matching_column

logger = logging.getLogger(__name__)


class UnsupportedFileException(Exception):
	"""File type was not supported."""


def read_dataframe_pickle(path: PurePath | str, **tqdm_kwargs) -> pandas.DataFrame:
	"""Reads a pickled DataFrame from a file path, displaying a progress bar for long files.

	Raises:
		TypeError: If the pickle file does not actually contain a DataFrame.
	"""
	if not isinstance(path, Path):
		path = Path(path)
	size = path.stat().st_size
	desc = tqdm_kwargs.pop('desc', f'Reading {path}')
	leave = tqdm_kwargs.pop('leave', False)
	with (
		path.open('rb') as f,
		tqdm.wrapattr(
			f, 'read', total=size, bytes=True, leave=leave, desc=desc, **tqdm_kwargs
		) as t,
	):
		# Don't really need to use pandas.read_pickle here, but also don't really need not to
		obj = pandas.read_pickle(t)  # type:ignore[blah] #supposedly, the wrapattr stream isn't entirely compatible with what pandas.read_pickle (or pickle.load) wants, but it's fine
	if not isinstance(obj, pandas.DataFrame):
		raise TypeError(f'Unpickled object was {type(obj)}, DataFrame expected')
	return obj


async def read_dataframe_pickle_async(path: PurePath | str, **tqdm_kwargs) -> pandas.DataFrame:
	"""Reads a pickled DataFrame from a file path in a separate thread, displaying a progress bar for long files."""
	# Could use aiofiles, but eh
	return await asyncio.to_thread(read_dataframe_pickle, path, **tqdm_kwargs)


def read_geodataframe(path: PurePath | str, *, use_tqdm: bool = True) -> geopandas.GeoDataFrame:
	"""Reads a GeoDataFrame from a path, which can be compressed using Zstandard."""
	if not isinstance(path, Path):
		path = Path(path)
	if path.suffix.lower() == '.zst':
		with (
			zstd.ZstdFile(path, 'r') as zst,
			tqdm.wrapattr(
				zst, 'read', path.stat().st_size, desc=f'Reading {path}', disable=not use_tqdm
			) as f,
		):
			# Getting the uncompressed size of zst would be nice but I don't think we can do that
			gdf = geopandas.read_file(f)
	elif use_tqdm:
		with (
			path.open('rb') as raw,
			tqdm.wrapattr(raw, 'read', path.stat().st_size, desc=f'Reading {path}') as f,
			warnings.catch_warnings(category=RuntimeWarning, action='ignore'),
		):
			# shut up nerd I don't care if it has a GPKG application_id or whatever (does this warning still get shown? Maybe not)
			gdf = geopandas.read_file(f)
	else:
		gdf = geopandas.read_file(path)
	return gdf


async def read_geodataframe_async(
	path: PurePath | str, *, use_tqdm: bool = True
) -> geopandas.GeoDataFrame:
	"""Reads a GeoDataFrame from a path in another thread, which can be compressed using Zstandard."""
	return await asyncio.to_thread(read_geodataframe, path, use_tqdm=use_tqdm)


def _geodataframe_to_normal_df(
	gdf: geopandas.GeoDataFrame,
	lat_col_name: Hashable = 'lat',
	lng_col_name: Hashable = 'lng',
	*,
	include_z: bool = False,
	insert_before: bool = True,
):
	only_has_points = all(isinstance(geom, Point) for geom in gdf.geometry.dropna())
	df = gdf.drop(columns=gdf.active_geometry_name)
	if only_has_points:
		coords = gdf.get_coordinates(include_z=include_z)
		a = [df, coords[coords.columns[::-1]]]  # generally we want lat before lng
		df = pandas.concat(reversed(a) if insert_before else a, axis='columns')
		return df.rename(columns={'x': lng_col_name, 'y': lat_col_name})
	return df


def geodataframe_to_csv(
	gdf: geopandas.GeoDataFrame,
	path: Path,
	lat_col_name: Hashable = 'lat',
	lng_col_name: Hashable = 'lng',
	*,
	include_z: bool = False,
	insert_before: bool = True,
	index: bool = True,
):
	"""Outputs a GeoDataFrame to CSV with lat and lng columns, instead of outputting as a WKT string."""
	df = _geodataframe_to_normal_df(
		gdf, lat_col_name, lng_col_name, include_z=include_z, insert_before=insert_before
	)
	df.to_csv(path, index=index)


dataframe_compressed_exts = {'gz', 'bz2', 'zip', 'xz', 'zst'}
csv_exts = {'csv', 'tsv'}
excel_exts = {'xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'}
pickle_exts = {'pickle', 'pkl'}
"""hmm there's nothing really that requires pickle data to be any sort of extension, but might as well be one of those"""
other_df_readers = {
	'feather': pandas.read_feather,
	'orc': pandas.read_orc,
	'parquet': pandas.read_parquet,
}
"""I don't know what these really do or if they work as expected because I don't use these file types but they exist and seem straightforward"""
dataframe_exts = csv_exts | excel_exts | pickle_exts | other_df_readers.keys()
html_exts = {'html', 'htm', 'xhtml'}


def output_dataframe(  # noqa: C901 #no u
	df: pandas.DataFrame,
	path: PurePath | str,
	force_ext: str | None = None,
	*,
	index: bool | None = None,
):
	"""Generically output a DataFrame to a path, automatically using to_csv/to_excel/etc as needed.

	Arguments:
		df: pandas DataFrame, if this is a GeoDataFrame you should probably use output_geodataframe instead as this will not convert it for you.
		path: File path to save to.
		force_ext: If this is specified (without a leading dot), pretend the suffix of path is this when detecting which save function to use, so you can e.g. pass "pickle" to save a pickle file with any extension since that does not really have a specific extension.
		index: Whether to output the index as a column or not (for CSV/Excel/etc). If None or unspecified, output the index if it is not a RangeIndex.
	"""
	if force_ext:
		ext = force_ext
	else:
		if not isinstance(path, PurePath):
			# We just need the suffix
			path = PurePath(path)
		# TODO: Handle compressed extensions, I guess we would take the extension before the compressed one
		ext = path.suffix[1:].lower()

	if index is None:
		index = not isinstance(df.index, pandas.RangeIndex)

	if ext == 'csv':
		df.to_csv(path, index=index)
	elif ext == 'tsv':
		df.to_csv(path, sep='\t', index=index)
	elif ext in excel_exts:
		df.to_excel(path, index=index)
		# Could autodetect which engine we are using and add some better formatting maybe
	elif ext in pickle_exts:
		df.to_pickle(path)
	elif ext in html_exts:
		df.to_html(path, index=index)
	elif ext == 'json':
		df.to_json(path, orient='index' if index else 'records')
	elif ext == 'md':
		df.to_markdown(path, index=index)
	else:
		raise UnsupportedFileException(f'Not sure how to output {ext}')


def output_geodataframe(
	gdf: geopandas.GeoDataFrame,
	path: PurePath | str,
	lat_col_name: Hashable = 'lat',
	lng_col_name: Hashable = 'lng',
	*,
	include_z: bool = False,
	insert_before: bool = True,
	index: bool = True,
	force_geojson_wgs84: bool = True,
):
	"""Outputs a GeoDataFrame automatically to the right format depending on the extension of `path`. `lat_col_name`, `lng_col_name`, `include_z`, `insert_before`, `index` are only used when outputting to a non-geographical format like csv/ods/etc."""
	# TODO: I guess you might want to handle compressed extensions
	if not isinstance(path, PurePath):
		# We just need the suffix
		path = PurePath(path)
	ext = path.suffix[1:].lower()
	if ext == 'csv':
		df = _geodataframe_to_normal_df(
			gdf, lat_col_name, lng_col_name, include_z=include_z, insert_before=insert_before
		)
		df.to_csv(path, index=index)
	elif ext in excel_exts:
		df = _geodataframe_to_normal_df(
			gdf, lat_col_name, lng_col_name, include_z=include_z, insert_before=insert_before
		)
		df.to_excel(path, index=index)
	elif ext in pickle_exts:
		gdf.to_pickle(path)
	else:
		if force_geojson_wgs84 and ext == 'geojson':
			gdf = gdf.to_crs('wgs84')
		gdf.to_file(path)


def read_dataframe(
	path: PurePath | str,
	ext: str | None = None,
	csv_encoding: str = 'utf-8',
	csv_sep: str | None = None,
	sheet_name: int | str = 0,
	*,
	has_header: bool | None = None,
) -> pandas.DataFrame:
	"""Loads a DataFrame from a path, using the right loader for the file extension.

	Arguments:
		path: File path
		ext: Override file format, or leave as None to autodetect from extension
		csv_encoding: If csv, text encoding to use
		csv_sep: If csv, field separator to use, defaults to comma (or tab if extension is tsv)
		sheet_name: If Excel, sheet name/index to use
		has_header: If csv/Excel, the data has column headers, defaults to infer (or True for Excel)

	Raises:
		UnsupportedFileException: If not a known file type.
	"""
	if not isinstance(path, PurePath):
		path = PurePath(path)
	if not ext:
		suffixes = path.suffixes
		ext = suffixes[-1][1:].lower()
		if ext in dataframe_compressed_exts:
			# not sure if read_csv/read_excel like compressed files, oh well
			ext = suffixes[-2][1:].lower()

	if ext in csv_exts:
		csv_sep = csv_sep or ('\t' if ext == 'tsv' else ',')
		# @.@ argh why
		if has_header is None:
			header = 'infer'
		elif has_header:
			header = 1
		else:
			header = None
		return pandas.read_csv(path, sep=csv_sep, encoding=csv_encoding, header=header)
	if ext in excel_exts:
		header_row = 0 if has_header in {True, None} else None
		# @.@ is there a better way to write that…
		return pandas.read_excel(path, sheet_name, header=header_row)
	if ext in pickle_exts:
		return read_dataframe_pickle(path)

	if ext in other_df_readers:
		return other_df_readers[ext](path)
	# TODO: fwf (as .txt maybe?), hdf (.h5), html, json, sas (.sas7bdat, whatever that is), spss (spss_data.sav? I dunno), stata (.dta), xml probably
	raise UnsupportedFileException(f'Unknown extension: {ext}')


async def read_dataframe_async(
	path: PurePath | str,
	ext: str | None = None,
	csv_encoding: str = 'utf-8',
	csv_sep: str | None = None,
	sheet_name: int | str = 0,
	*,
	has_header: bool | None = None,
) -> pandas.DataFrame:
	"""Loads a DataFrame from a path, using the right loader for the file extension.

	Arguments:
		path: File path
		ext: Override file format, or leave as None to autodetect from extension
		csv_encoding: If csv, text encoding to use
		csv_sep: If csv, field separator to use, defaults to comma (or tab if extension is tsv)
		sheet_name: If Excel, sheet name/index to use
		has_header: If csv/Excel, the data has column headers, defaults to infer (or True for Excel)
	"""
	return await asyncio.to_thread(
		read_dataframe, path, ext, csv_encoding, csv_sep, sheet_name, has_header=has_header
	)


latitude_column_names = {'lat', 'latitude', 'Latitude', 'y', 0}
longitude_column_names = {'lng', 'lon', 'longitude', 'Longitude', 'x', 1}


def _load_df_as_points(
	path: PurePath | str,
	latitude_column_name: Hashable | None = None,
	longitude_column_name: Hashable | None = None,
	crs: Any = 'wgs84',
	ext: str | None = None,
	*,
	has_header: bool | None = None,
	keep_lnglat_cols: bool = False,
) -> geopandas.GeoDataFrame:
	df = read_dataframe(path, ext, has_header=has_header)
	latitude_column_name = latitude_column_name or find_first_matching_column(
		df, latitude_column_names
	)
	longitude_column_name = longitude_column_name or find_first_matching_column(
		df, longitude_column_names
	)

	if latitude_column_name is None or longitude_column_name is None:
		raise UnsupportedFileException(
			f'This file does not seem to contain any columns for points (found: {df.columns})'
		)
	lat = df[latitude_column_name]
	lng = df[longitude_column_name]

	if not keep_lnglat_cols:
		df = df.drop(columns=[latitude_column_name, longitude_column_name])
	geometry = geopandas.points_from_xy(lng, lat, crs=crs)
	return geopandas.GeoDataFrame(df, geometry=geometry)


def load_points(
	path: PurePath | str,
	latitude_column_name: Hashable | None = None,
	longitude_column_name: Hashable | None = None,
	crs: Any = 'wgs84',
	ext: str | None = None,
	*,
	has_header: bool | None = None,
	keep_lnglat_cols: bool = False,
	use_tqdm: bool = True,
) -> geopandas.GeoDataFrame:
	"""Loads a file containing coordinates as a GeoDataFrame, either as a DataFrame (csv/Excel/pickle/etc) containing longitude and latitude columns, or a file directly supported by geopandas."""
	if not isinstance(path, PurePath):
		path = Path(path)
	if not ext:
		suffixes = path.suffixes
		ext = suffixes[-1][1:].lower()
		if ext in dataframe_compressed_exts:
			ext = suffixes[-2][1:].lower()
	if ext not in dataframe_exts:
		return read_geodataframe(path, use_tqdm=use_tqdm)
	return _load_df_as_points(
		path,
		latitude_column_name,
		longitude_column_name,
		crs,
		ext,
		has_header=has_header,
		keep_lnglat_cols=keep_lnglat_cols,
	)


async def load_points_async(
	path: PurePath | str,
	latitude_column_name: Hashable | None = None,
	longitude_column_name: Hashable | None = None,
	crs: Any = 'wgs84',
	ext: str | None = None,
	*,
	has_header: bool | None = None,
	keep_lnglat_cols: bool = False,
	use_tqdm: bool = True,
) -> geopandas.GeoDataFrame:
	"""Loads a file containing coordinates as a GeoDataFrame asynchronously, either as a DataFrame (csv/Excel/pickle/etc) containing longitude and latitude columns, or a file directly supported by geopandas"""
	if not isinstance(path, PurePath):
		path = Path(path)
	if not ext:
		suffixes = path.suffixes
		ext = suffixes[-1][1:].lower()
		if ext in dataframe_compressed_exts:
			ext = suffixes[-2][1:].lower()
	if ext not in dataframe_exts:
		return await read_geodataframe_async(path, use_tqdm=use_tqdm)
	return await asyncio.to_thread(
		_load_df_as_points,
		path,
		latitude_column_name,
		longitude_column_name,
		crs,
		ext,
		has_header=has_header,
		keep_lnglat_cols=keep_lnglat_cols,
	)


def geometry_to_file(
	path: str | PurePath, geom: BaseGeometry, crs: Any = 'wgs84', *, explode_multipart: bool = True
):
	"""Exports shapely geometry objects to a file containing just that geometry."""
	# TODO: Pretty print if output format is geojson
	if explode_multipart and isinstance(geom, BaseMultipartGeometry):
		data = list(geom.geoms)
	else:
		data = [geom]
	gs = geopandas.GeoSeries(data, crs=crs)
	gs.to_file(path)


async def geometry_to_file_async(
	path: str | PurePath, geom: BaseGeometry, crs: Any = 'wgs84', *, explode_multipart: bool = True
):
	"""Exports shapely geometry objects to a file containing just that geometry."""
	# TODO: Pretty print if output format is geojson
	if explode_multipart and isinstance(geom, BaseMultipartGeometry):
		data = list(geom.geoms)
	else:
		data = [geom]
	gs = geopandas.GeoSeries(data, crs=crs)
	await asyncio.to_thread(gs.to_file, path)


known_geo_exts = {'geojson', 'gpkg', 'shp'}
"""A subset of file extensions that we can be reasonably sure will be supported."""


def maybe_load_geodataframe(
	path: str | PurePath, *, use_tqdm: bool = True
) -> geopandas.GeoDataFrame | None:
	"""Attemps to load a GeoDataFrame from a path, but returns None if not supported."""
	try:
		from pyogrio.errors import DataSourceError as UnsupportedError  # noqa: PLC0415
	except ImportError:
		try:
			from fiona.errors import DriverError as UnsupportedError  # noqa: PLC0415
		except ImportError:
			logger.warning(
				'You have neither pyogrio or fiona installed, so nothing is loadable anyway'
			)
			return None
	try:
		return read_geodataframe(path, use_tqdm=use_tqdm)
	except UnsupportedError:
		return None
