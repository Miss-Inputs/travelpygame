"""Functions to access Morphior's site to get all.geojson, because it seemed best to put them in a different module."""

from io import StringIO
from typing import TYPE_CHECKING

import geopandas

from .util.web import get_text

if TYPE_CHECKING:
	from aiohttp import ClientSession, ClientTimeout


async def get_morphior_all_submissions_version(
	session: 'ClientSession | None' = None, client_timeout: 'ClientTimeout | float | None' = 60.0
) -> str:
	"""Returns some plain text indicating a version ID for all.geojson. We should use this somehow, but I haven't thought of an ideal way to do that"""
	url = 'https://tpg.marsmathis.com/headsup/data/mymaps/all.version'
	return await get_text(url, session, client_timeout)


async def get_morphior_all_submissions_json(
	session: 'ClientSession | None' = None, client_timeout: float | None = 60
) -> str:
	"""Returns json as plain text."""
	url = 'https://tpg.marsmathis.com/headsup/data/mymaps/all.geojson'
	"""Endpoint with v=<version> parameter"""
	return await get_text(url, session, client_timeout)


async def get_morphior_all_submissions(
	session: 'ClientSession | None' = None, client_timeout: float | None = 60
) -> geopandas.GeoDataFrame:
	"""Returns all.geojson (containing submissions across spinoffs) as a GeoDataFrame. Currently has these columns:
	geometry
	name: Username
	description: Usually empty string, or the description as entered in the submission tracker
	styleUrl: Google Maps style URL
	Coordinates: Coordinates as comma-separated decimal degrees
	source: "mymaps"
	map_id: Google My Maps ID
	layer: Seems to be always an empty string for now
	title: Username
	username: Username
	"""
	json_text = await get_morphior_all_submissions_json(session, client_timeout)
	stringy = StringIO(json_text)
	gdf = geopandas.read_file(stringy)
	for col in ('description', 'layer'):
		gdf[col] = gdf[col].replace('', None)
	return gdf.dropna(axis='columns', how='all')
