from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from enum import IntEnum
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy
import pandas

from travelpygame.util.distance import geod_distance_and_bearing, haversine_distance

if TYPE_CHECKING:
	from .submissions import Round


def tpg_score(distances: 'pandas.Series', *, allow_negative: bool = False):
	"""
	Computes the score for a whole round of TPG. Note: Not complete yet, this does not take ties into account.

	Arguments:
		distances: Distances in kilometres for each round.
		allow_negative: Allow distance scores to be negative, if false (default) give a score of 0 if distance is greater than 20_000km, which is impossible except for exact antipodes by a few km, but just for completeness/symmetry with custom_tpg_score
	"""
	distance_scores = 0.25 * (20_000 - distances)
	if not allow_negative:
		distance_scores = distance_scores.clip(0)

	distance_ranks = distances.rank(method='min', ascending=True)
	players_beaten = distances.size - distances.rank(method='max', ascending=True)
	players_beaten_scores = 5000 * (players_beaten / (distances.size - 1))
	scores = distance_scores + players_beaten_scores
	bonus = distance_ranks.map({1: 3000, 2: 2000, 3: 1000})
	# TODO: Should actually just pass in the fivek column
	bonus.loc[distances <= 0.1] = 5000
	scores += bonus.fillna(0)
	scores.loc[distances >= 19_995] = 5000  # Antipode 5K
	for _, group in scores.groupby(distance_ranks, sort=False):
		# where distance is tied, all players in that tie receive the average of what the points would be
		scores.loc[group.index] = group.mean()
	return scores.round(2)

def score_distances(
	distances: pandas.Series,
	is_5k: pandas.Series,
	world_distance_km: float = 5_000.0,
	fivek_score: float | None = 7_500.0,
	*,
	clip_negative: bool = True,
	round_to: int | None = 2,
):
	# TODO: Take into account antipode 5Ks and ties
	n = distances.size
	world_distance = world_distance_km * 1_000
	distance_scores = (world_distance - distances) / 1_000
	if clip_negative:
		distance_scores = distance_scores.clip(0)

	players_beaten = n - distances.rank(method='max', ascending=True)
	players_beaten_scores = 5000 * (players_beaten / (n - 1))
	scores = (distance_scores + players_beaten_scores) / 2
	if fivek_score:
		scores[is_5k] = fivek_score
	return scores if round_to is None else scores.round(round_to)


def score_round(
	round_: 'Round',
	world_distance: float = 5_000.0,
	fivek_score: float | None = 7_500.0,
	fivek_threshold: float | None = 0.1,
	*,
	use_haversine: bool = True,
	clip_negative: bool = True,
) -> 'Round':
	n = len(round_.submissions)
	subs = pandas.DataFrame([s.model_dump() for s in round_.submissions])

	lats = subs['latitude'].to_numpy()
	lngs = subs['longitude'].to_numpy()
	target_lat = numpy.repeat(round_.latitude, n)
	target_lng = numpy.repeat(round_.longitude, n)
	if subs['distance'].hasnans:
		#We don't have to recalc distance if we somehow have distance (but not score) for every submission, but if any of them don't then we need to recalc anyway
		if use_haversine:
			distances = haversine_distance(lats, lngs, target_lat, target_lng)
			# TODO: Option to calc geod distance/bearing anyway, just for funsies
		else:
			distances, _bearings = geod_distance_and_bearing(target_lat, target_lng, lats, lngs)
		subs['distance'] = distances

	if fivek_threshold is not None:
		within_threshold = subs['distance'] <= fivek_threshold
		subs['is_5k'] = subs['is_5k'].astype('boolean').fillna(within_threshold)

	scores = score_distances(
		subs['distance'], subs['is_5k'], world_distance, fivek_score, clip_negative=clip_negative
	)
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
