from collections import Counter
from collections.abc import Iterable

import geopandas
from shapely import Point

from .classes import PlayerName, Round, Season

_CombinationSource = Round | Season | tuple[PlayerName, float, float]


def _add_combined_subs(
	d: dict[PlayerName, Counter[Point]],
	source: _CombinationSource,
	aliases: dict[str, PlayerName],
	rounding: int,
):
	if isinstance(source, Round):
		for sub in source.submissions:
			_add_combined_subs(d, (sub.name, sub.latitude, sub.longitude), aliases, rounding)
	elif isinstance(source, Season):
		for roundyboi in source.rounds:
			_add_combined_subs(d, roundyboi, aliases, rounding)
	else:
		player, lat, lng = source
		player = aliases.get(player, player)
		point = Point(round(lng, rounding), round(lat, rounding))
		d.setdefault(player, Counter())[point] += 1


def combine_player_submissions(
	sources: Iterable[_CombinationSource], aliases: dict[str, PlayerName] | None, rounding: int = 6
) -> dict[PlayerName, geopandas.GeoDataFrame]:
	"""Combines rounds with submissions/mappings of player name + coordinates into one mapping of submissions by player.

	Arguments:
		aliases: Optional mapping of {raw player name -> player name}, for spelling variations in trackers and such.
		rounding: Number of decimal places to round to.

	Returns:
	        dict containing player submissions by name."""
	# TODO: Probably should ensure player name is case insensitive or whatnot, though just need to ensure it still ends up the correct way

	combined: dict[PlayerName, Counter[Point]] = {}
	for source in sources:
		_add_combined_subs(combined, source, aliases or {}, rounding)
	frames: dict[PlayerName, geopandas.GeoDataFrame] = {}
	for player, points in frames.items():
		point_counts = [{'point': point, 'count': count} for point, count in points.items()]
		frames[player] = geopandas.GeoDataFrame(point_counts, geometry='point', crs='wgs84')
	return frames
