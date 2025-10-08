from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import IntEnum
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy
import pandas

from .util.distance import geod_distance_and_bearing, haversine_distance

if TYPE_CHECKING:
	from .tpg_data import Round


@dataclass
class ScoringOptions:
	"""Different ways to score TPG. This is not an exhaustive set of configurations, but it's what we use at the moment"""

	fivek_flat_score: float | None
	"""If not None, 5Ks have a constant score of this"""
	fivek_bonus: float | None
	"""If not None, add this amount to the score of any submission which is a 5K (note that this would be in conjunction with rank_bonus)"""
	rank_bonuses: dict[int, float] | None
	"""If not none, add amounts to the score of players receiving a certain rank, e.g. main TPG uses {1: 3000, 2: 2000, 1: 1000}"""
	antipode_5k_flat_score: float | None
	"""If not None, antipode 5Ks have a constant score of this"""
	world_distance_km: float = 20_000
	"""Maximum distance (size of what is considered the entire world), usually simplified as 20,000 for world TPG or 5,000 for most regional TPGs"""
	round_to: int | None = 2
	"""Round score to this many decimal places, or None to not do that"""
	distance_divisor: float | None = None
	"""If not None, divide distance by this, and then subtract from (world_distance_km / 4) (this is basically just for main TPG)"""
	clip_negative: bool = True
	"""Sets negative distance scores to 0, you probably want this in regional TPGs to be nice to players who are outside your region but submit anyway for the love of the game (so they can get points from rank instead)"""
	average_distance_and_rank: bool = True
	"""If true, score is (distance score + rank score) / 2, if false (as with main TPG), just add the two score parts together"""


main_tpg_scoring = ScoringOptions(
	None,
	2000,
	{1: 3000, 2: 2000, 3: 1000},
	10_000,
	distance_divisor=4.003006,
	average_distance_and_rank=False,
)


def score_distances(
	distances: pandas.Series,
	is_5k: pandas.Series,
	is_antipode_5k: pandas.Series,
	options: ScoringOptions,
):
	n = distances.size

	if options.distance_divisor:
		distance_scores = (options.world_distance_km / 4) - (
			(distances / 1_000) / options.distance_divisor
		)
	else:
		world_distance = options.world_distance_km * 1_000
		distance_scores = (world_distance - distances) / 1_000
	if options.clip_negative:
		distance_scores = distance_scores.clip(0)

	players_beaten = n - distances.rank(method='max', ascending=True)
	players_beaten_scores = 5000 * (players_beaten / (n - 1))

	scores = distance_scores + players_beaten_scores
	if options.average_distance_and_rank:
		scores /= 2

	if options.rank_bonuses:
		ranks = scores.rank(method='dense', ascending=False)
		for rank, bonus in options.rank_bonuses.items():
			scores[ranks == rank] += bonus

	if options.fivek_flat_score is not None:
		scores[is_5k] = options.fivek_flat_score
	elif options.fivek_bonus is not None:
		scores[is_5k] += options.fivek_bonus
	if options.antipode_5k_flat_score is not None:
		scores[is_antipode_5k] = options.antipode_5k_flat_score
	return scores if options.round_to is None else scores.round(options.round_to)


def score_round(
	round_: 'Round',
	options: ScoringOptions,
	fivek_threshold: float | None = 0.1,
	*,
	use_haversine: bool = True,
) -> 'Round':
	# TODO: Take into account ties (all players within a group of is_tie have their scores averaged out)
	n = len(round_.submissions)
	if n == 0:
		return round_
	subs = pandas.DataFrame([s.model_dump() for s in round_.submissions])

	lats = subs['latitude'].to_numpy()
	lngs = subs['longitude'].to_numpy()
	target_lat = numpy.repeat(round_.latitude, n)
	target_lng = numpy.repeat(round_.longitude, n)
	if subs['distance'].hasnans:
		# We don't have to recalc distance if we somehow have distance (but not score) for every submission, but if any of them don't then we need to recalc anyway
		if use_haversine:
			distances = haversine_distance(lats, lngs, target_lat, target_lng)
			# TODO: Option to calc geod distance/bearing anyway, just for funsies
		else:
			distances, _bearings = geod_distance_and_bearing(target_lat, target_lng, lats, lngs)
		subs['distance'] = distances

	if fivek_threshold is not None:
		within_threshold = subs['distance'] <= fivek_threshold
		subs['is_5k'] = subs['is_5k'].astype('boolean').fillna(within_threshold)

	is_antipode_5k = subs['is_antipode_5k'].astype('boolean').fillna(value=False)
	scores = score_distances(subs['distance'], subs['is_5k'], is_antipode_5k, options)
	ranks = scores.rank(ascending=False).astype(int)
	scored_subs = [
		s.model_copy(
			update={
				'rank': ranks.iloc[i].item(),  # pyright: ignore[reportAttributeAccessIssue]
				'score': scores.iloc[i].item(),
				'distance': subs['distance'].iloc[i].item(),
				'is_5k': subs['is_5k'].iloc[i].item(),
			}
		)
		for i, s in enumerate(round_.submissions)
	]
	scored_subs.sort(key=attrgetter('rank'))
	return round_.model_copy(update={'submissions': scored_subs})


class Medal(IntEnum):
	"""Medals that are worth points for 1st/2nd/3rd place in a round."""

	Gold = 3
	Silver = 2
	Bronze = 1


def _count_medals(medals: Mapping[str, Collection[Medal]]):
	"""Tallies medals from podium placements.

	Arguments:
		medals: {player name: [all medals obtained in the season]}

	Returns:
		DataFrame, indexed by player name, with columns for counts of each medal and a "Medal Score" column for total medal points (with gold medals being 3 points, silver medals worth 2, etc) that it is sorted by
	"""

	counts: dict[str, dict[str, int]] = {medal: {} for medal in Medal._member_names_}
	"""{medal type: {player name: amount of times this medal was won}}"""
	points: dict[str, int] = {}
	"""{player name: total points of all medals}"""

	for player_name, player_medals in medals.items():
		counter = Counter(player_medals)
		for medal, count in counter.items():
			counts[medal.name][player_name] = count
		points[player_name] = sum(player_medals)

	df = pandas.DataFrame(counts, dtype='Int64')
	df.index.name = 'Player'
	df = df.fillna(0)
	df['Medal Score'] = points
	return df.sort_values('Medal Score', ascending=False)


def _add_totals(df: pandas.DataFrame, *, ascending: bool):
	df.insert(0, 'Total', df.sum(axis='columns'))
	df.insert(1, 'Average', df['Total'] / df.columns.size)
	return df.sort_values('Total', ascending=ascending)


def make_leaderboards(rounds: list['Round']):
	"""Returns tuple of (points leaderboard, distance leaderboard, medal leaderboard)"""
	# name: {player: score}
	points: defaultdict[str, dict[str, float]] = defaultdict(dict)
	distances: defaultdict[str, dict[str, float]] = defaultdict(dict)
	# player: all medals
	medals: defaultdict[str, list[Medal]] = defaultdict(list)

	for r in rounds:
		name = r.name or f'Round {r.number}'
		for sub in r.submissions:
			if sub.score is None or sub.distance is None:
				raise ValueError('Submissions must be scored to make leaderboards')
			points[name][sub.name] = sub.score
			distances[name][sub.name] = sub.distance / 1_000
			if sub.rank and sub.rank <= 3:
				medals[sub.name].append(Medal(4 - sub.rank))

	points_leaderboard = pandas.DataFrame(points)
	points_leaderboard.index.name = 'Points'
	points_leaderboard = _add_totals(points_leaderboard, ascending=False)

	distance_leaderboard = pandas.DataFrame(distances).dropna()
	distance_leaderboard.index.name = 'Distance'
	distance_leaderboard = _add_totals(distance_leaderboard, ascending=True)

	medals_leaderboard = _count_medals(medals)
	return points_leaderboard, distance_leaderboard, medals_leaderboard
