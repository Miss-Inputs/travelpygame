from typing import Any, Literal

import aiohttp
import backoff
import pydantic_core
from async_lru import alru_cache
from pydantic import BaseModel


class NominatimReverseJSONv2(BaseModel, extra='allow'):
	"""JSON v2 format, used by reverse_geocode_address."""

	place_id: int
	"""OSM place ID for the nearest object found, which is the address we use."""
	licence: str
	"""Copyright string"""
	osm_type: str
	"""node, way, etc"""
	osm_id: int
	lat: float
	lon: float
	category: str
	"""man_made, etc"""
	type: str
	place_rank: int
	importance: float
	addresstype: str
	name: str
	"""Name of the node"""
	display_name: str
	"""Full formatted address."""
	boundingbox: tuple[float, float, float, float]
	"""min lat, max lat, min long, max long"""


class NominatimReverseJSONv2WithAddressDetails(NominatimReverseJSONv2):
	address: dict[str, str]


class NominatimGeocoding(BaseModel, extra='allow'):
	version: str
	"""0.1.0"""
	attribution: str
	"""Copyright string"""
	licence: str
	"""ODbL"""
	query: Literal['']


class NominatimGeocodingProperties(BaseModel, extra='allow'):
	"""Could have basically anything in it. Used by reverse_geocode_components."""

	place_id: int
	osm_type: str
	osm_id: int
	osm_key: str
	osm_value: str
	osm_type: str
	accuracy: int
	label: str
	"""Fully formatted address."""
	name: str | None = None
	postcode: str | None = None
	street: str | None = None
	district: str | None = None
	city: str | None = None
	state: str | None = None
	country: str | None = None
	"""Should _usually_ be there unless you're in a strange location (e.g. Antarctica)"""
	country_code: str | None = None
	admin: dict[str, str]
	"""keys: level9, level7, level4, etc"""


class NominatimProperties(BaseModel, extra='forbid'):
	"""huh? What a pointless object"""

	geocoding: NominatimGeocodingProperties


class NominatimFeature(BaseModel, extra='forbid'):
	type: Literal['Feature']
	properties: NominatimProperties
	geometry: Any
	"""type = point and coordinates or whatever, don't care"""


class NominatimReverseGeocodeJSON(BaseModel, extra='allow'):
	type: Literal['FeatureCollection']
	geocoding: NominatimGeocoding
	features: list[NominatimFeature]


class GeocodeError(Exception):
	pass


DEFAULT_ENDPOINT = 'https://nominatim.geocoding.ai/reverse'
"""It seems like this just redirects to normal OSM now? So maybe that's not good"""

@alru_cache
@backoff.on_exception(backoff.expo, aiohttp.ClientResponseError)
async def reverse_geocode_address(
	lat: float,
	lng: float,
	session: aiohttp.ClientSession,
	lang: str = 'en',
	endpoint: str = DEFAULT_ENDPOINT,
	request_timeout: int = 30,
) -> str | None:
	"""Finds an address for a point using asynchronous requests.

	Raises:
		GeocodeError: If some weird error happens.

	Arguments:
		lat: Latitude of point in WGS84.
		lng: Longitude of point in WGS84.
		session: Optional requests.Session if you have one, otherwise does not use a session. Recommended if you are using this in a loop, etc.
		request_timeout: Request timeout in seconds, defaults to 30 seconds.

	Returns:
		Address as string, or None if nothing could be found.
	"""
	params = {
		'lat': lat,
		'lon': lng,
		'format': 'jsonv2',
		'addressdetails': 0,
		'accept-language': lang,
	}
	async with session.get(
		endpoint, params=params, timeout=aiohttp.ClientTimeout(request_timeout)
	) as response:
		response.raise_for_status()
		text = await response.text()

	j = pydantic_core.from_json(text)
	error = j.get('error')
	if error == 'Unable to geocode':
		return None
	if error:
		raise GeocodeError(error)
	return NominatimReverseJSONv2.model_validate(j).display_name


@backoff.on_exception(backoff.expo, aiohttp.ClientResponseError)
async def reverse_geocode_components(
	lat: float,
	lng: float,
	session: aiohttp.ClientSession,
	lang: str = 'en',
	endpoint: str = DEFAULT_ENDPOINT,
	request_timeout: int = 10,
) -> NominatimReverseGeocodeJSON | None:
	"""Returns individual address components instead of just a string.

	Raises:
		GeocodeError: If some weird error happens that isn't just 'unable to geocode'
	"""
	params = {
		'lat': lat,
		'lon': lng,
		'format': 'geocodejson',
		'addressdetails': 1,
		'accept-language': lang,
	}

	async with session.get(
		endpoint, params=params, timeout=aiohttp.ClientTimeout(request_timeout)
	) as response:
		response.raise_for_status()
		text = await response.text()
	j = pydantic_core.from_json(text)
	error = j.get('error')
	if error == 'Unable to geocode':
		return None
	if error:
		raise GeocodeError(error)
	return NominatimReverseGeocodeJSON.model_validate(j)
