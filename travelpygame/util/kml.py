"""Parser for KML files exported from Google My Maps"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from shapely import Point

logger = logging.getLogger(__name__)


@dataclass
class Placemark:
	name: str
	description: str | None
	style: str | None
	point: Point


class KMLError(Exception):
	"""Something unexpected with the KML"""


@dataclass
class SubmissionTrackerRound:
	name: str
	target: Point
	antipode: Point | None
	submissions: list[Placemark]


@dataclass
class SubmissionTracker:
	rounds: list[SubmissionTrackerRound]


def _parse_placemark(placemark: ElementTree.Element):
	name = placemark.findtext('{http://www.opengis.net/kml/2.2}name') or ''
	description = placemark.findtext('{http://www.opengis.net/kml/2.2}description')
	style = placemark.findtext('{http://www.opengis.net/kml/2.2}styleUrl')
	point = placemark.find('{http://www.opengis.net/kml/2.2}Point')
	if point is None:
		raise KMLError(f"{name}'s submission is pointless!")

	coordinates_text = point.findtext('{http://www.opengis.net/kml/2.2}coordinates')
	if not coordinates_text:
		raise KMLError(f"{name}'s submission has a point with no coordinates?")
	coordinates = coordinates_text.strip().split(',')
	# Third element is probably elevation but is unused and always 0 for TPG
	lng = float(coordinates[0])
	lat = float(coordinates[1])
	point = Point(lng, lat)
	return Placemark(name, description, style, point)


def _parse_folder(folder: ElementTree.Element, *, include_antipode: bool = False):
	name = folder.findtext('{http://www.opengis.net/kml/2.2}name') or ''

	placemark_iter = folder.iter('{http://www.opengis.net/kml/2.2}Placemark')
	# Assume the first two elements are the round location itself, and optionally the antipode
	target_placemark_xml = next(placemark_iter)
	target_placemark = _parse_placemark(target_placemark_xml)
	target = target_placemark.point

	if include_antipode:
		antipode_placemark_xml = next(placemark_iter)
		antipode_placemark = _parse_placemark(antipode_placemark_xml)
		antipode = antipode_placemark.point
	else:
		antipode = None

	submissions = []
	for placemark in placemark_iter:
		try:
			submissions.append(_parse_placemark(placemark))
		except KMLError:
			logger.exception('Unexpected submission error:')
			continue

	return SubmissionTrackerRound(name, target, antipode, submissions)


def _parse_kmz(path: Path):
	with ZipFile(path, 'r') as z, z.open('doc.kml') as f:
		return ElementTree.parse(f)


def _parse_kml_rounds(path: Path | ElementTree.ElementTree, *, include_antipode: bool = False):
	if isinstance(path, ElementTree.ElementTree):
		tree = path
	elif path.suffix[1:].lower() == 'kmz':
		tree = _parse_kmz(path)
	else:
		tree = ElementTree.parse(path)

	doc = tree.find('{http://www.opengis.net/kml/2.2}Document', {})
	if doc is None:
		raise KMLError(f'{path} has no document')

	return [
		_parse_folder(folder, include_antipode=include_antipode)
		for folder in doc.iter('{http://www.opengis.net/kml/2.2}Folder')
	]


def parse_submission_kml(
	path: Path | ElementTree.ElementTree | Sequence[Path], *, include_antipode: bool = False
):
	"""Parses .kml file(s) from submission trackers. path can be multiple paths, in the likely event there are more than 10 rounds so a single one cannot store them all due to Google My Maps limits.
	
	Arguments:
		path: Path to file(s), or existing parsed XML.
		include_antipode: If true, the second item in every layer is the antipode, and not a submission."""
	if isinstance(path, Sequence):
		rounds = list(
			chain.from_iterable(
				_parse_kml_rounds(p, include_antipode=include_antipode) for p in path
			)
		)
	else:
		rounds = _parse_kml_rounds(path, include_antipode=include_antipode)

	return SubmissionTracker(rounds)
