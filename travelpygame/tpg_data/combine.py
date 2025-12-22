from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

import geopandas
from shapely import Point

from .classes import PlayerName, PlayerUsername, Round, Season

PointCounters = dict[PlayerUsername, Counter[Point]]
_PlayerWithCoords = tuple[PlayerUsername, float, float]
_CombinationSource = (
	Round
	| Season
	| Mapping[PlayerUsername, Collection[tuple[float, float]]]
	# I would have preferred to just do Iterable[_PlayerWithCoords] but that causes shenanigans in _add_combined_subs because type hinters see Mapping[_PlayerWithCoords, Unknown] as valid in that instanceof branch
	| Iterator[_PlayerWithCoords]
	| Sequence[_PlayerWithCoords]
)


def _add_combined_subs(
	d: PointCounters,
	source: _CombinationSource,
	aliases: Mapping[PlayerName, PlayerUsername],
	rounding: int,
):
	if isinstance(source, Round):
		_add_combined_subs(
			d,
			((sub.username or sub.name, sub.latitude, sub.longitude) for sub in source.submissions),
			aliases,
			rounding,
		)
	elif isinstance(source, Season):
		for roundyboi in source.rounds:
			_add_combined_subs(d, roundyboi, aliases, rounding)
	elif isinstance(source, Mapping):
		for player, points in source.items():
			_add_combined_subs(d, ((player, lat, lng) for lat, lng in points), aliases, rounding)
	else:
		for player, lat, lng in source:
			player = aliases.get(player, player)
			point = Point(round(lng, rounding), round(lat, rounding))
			d.setdefault(player, Counter())[point] += 1


def convert_point_counters(d: PointCounters) -> dict[PlayerUsername, geopandas.GeoDataFrame]:
	frames: dict[PlayerUsername, geopandas.GeoDataFrame] = {}
	for player, points in d.items():
		point_counts = [{'point': point, 'count': count} for point, count in points.items()]
		frames[player] = geopandas.GeoDataFrame(point_counts, geometry='point', crs='wgs84')
	return frames


def combine_point_counters(
	counters: Iterable[PointCounters],
) -> dict[PlayerUsername, geopandas.GeoDataFrame]:
	combined: PointCounters = {}
	for counter in counters:
		for player, points in counter.items():
			combined[player].update(points)
	return convert_point_counters(combined)


def combine_player_submissions(
	sources: Iterable[_CombinationSource],
	aliases: Mapping[PlayerName, PlayerUsername] | None = None,
	rounding: int = 6,
) -> PointCounters:
	"""Combines rounds with submissions/mappings of player name + coordinates into one mapping of submissions by player.

	Arguments:
		sources: Iterable of Round, Season, dict of {player name -> collection of (lat, lng) tuples}, or iterator/sequence of (player name, lat, lng) tuples
		aliases: Optional mapping of {raw player name -> player name}, for spelling variations in trackers and such.
		rounding: Number of decimal places to round to.

	Returns:
	        dict containing player submissions by name."""
	# TODO: Probably should ensure player name is case insensitive or whatnot, though just need to ensure it still ends up the correct way
	combined: PointCounters = {}
	for source in sources:
		_add_combined_subs(combined, source, aliases or {}, rounding)
	return combined


def combine_player_submissions_to_point_sets(
	sources: Iterable[_CombinationSource],
	aliases: Mapping[PlayerName, PlayerUsername] | None = None,
	rounding: int = 6,
) -> dict[PlayerUsername, geopandas.GeoDataFrame]:
	"""Combines rounds with submissions/mappings of player name + coordinates into one mapping of submissions by player.

	Arguments:
		aliases: Optional mapping of {raw player name -> player name}, for spelling variations in trackers and such.
		rounding: Number of decimal places to round to.

	Returns:
	        dict containing player submissions by name."""
	combined = combine_player_submissions(sources, aliases, rounding)
	return convert_point_counters(combined)
