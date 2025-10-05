"""Functions to simulate TPG seasons, playing out rounds as though everyone was there to submit their best pic."""

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from operator import itemgetter
from statistics import mean
from typing import Any

import pandas
import shapely
from geopandas import GeoSeries
from shapely import Point
from tqdm.auto import tqdm

from .best_pics import get_best_pic
from .scoring import ScoringOptions, main_tpg_scoring, score_round
from .tpg_data import Round, Submission, get_submissions_per_user

logger = logging.getLogger(__name__)


class SimulatedStrategy(Enum):
	Closest = auto()
	"""Simulated players will choose their closest pic. This is the sensible option and the default."""
	Furthest = auto()
	"""Simulated players will choose their furthest away pic, for whatever reason."""
	Random = auto()
	"""Simulated players will ignore the target and just randomly choose a random pic each time."""


@dataclass
class Simulation:
	rounds: dict[str, Point]
	round_order: dict[str, int] | None
	"""Order for each round, if desired, effectively sets the round number (i.e. lower numbers first)"""
	player_pics: dict[str, GeoSeries | Sequence[Point]]
	"""Locations for each user."""
	scoring: ScoringOptions
	strategy: SimulatedStrategy = SimulatedStrategy.Closest
	use_haversine: bool = True
	use_tqdm: bool = True

	def simulate_round(self, name: str, number: int, target: Point) -> Round:
		submissions: list[Submission] = []
		for player, pics in self.player_pics.items():
			if self.strategy == SimulatedStrategy.Random:
				if isinstance(pics, GeoSeries):
					desc, point = next(pics.sample(1).items())
					assert isinstance(point, Point), f'point was {type(point)}, expected Point'
				else:
					desc = None
					point = random.choice(pics)
				# We will just let distance be calculated later
				distance = None
			else:
				best_index, distance = get_best_pic(
					pics,
					target,
					use_haversine=self.use_haversine,
					reverse=self.strategy == SimulatedStrategy.Furthest,
				)
				desc = best_index if isinstance(pics, GeoSeries) else None
				point = pics[best_index]
				assert isinstance(point, Point), f'point was {type(point)}, expected Point'

			submissions.append(
				Submission(
					name=player,
					latitude=point.y,
					longitude=point.x,
					description=desc if isinstance(desc, str) else None,
					distance=distance,
				)
			)

		r = Round(
			name=name, number=number, latitude=target.y, longitude=target.x, submissions=submissions
		)
		return score_round(r, self.scoring, use_haversine=self.use_haversine)

	def simulate_rounds(self) -> list[Round]:
		items = self.rounds.items()
		round_order = self.round_order
		if round_order:
			items = [
				kv
				for _, kv in sorted(
					enumerate(items), key=lambda i_kv: round_order.get(i_kv[1][0], i_kv[0])
				)
			]
		if self.use_tqdm:
			rounds = []
			with tqdm(items, 'Simulating rounds', unit='round') as t:
				for i, (name, target) in enumerate(t, 1):
					t.set_postfix(round=name)
					rounds.append(self.simulate_round(name, i, target))
			return rounds
		return [self.simulate_round(name, i, target) for i, (name, target) in enumerate(items, 1)]


def simulate_existing_rounds(
	rounds: Collection[Round],
	scoring: ScoringOptions | None = None,
	strategy: SimulatedStrategy = SimulatedStrategy.Closest,
	*,
	use_haversine: bool = True,
) -> Simulation:
	pics = {
		player: shapely.points([(lng, lat) for lat, lng in latlngs]).tolist()
		for player, latlngs in get_submissions_per_user(rounds).items()
	}
	targets = {
		r.name or f'Round {r.number}': shapely.Point(r.longitude, r.latitude) for r in rounds
	}
	scoring = scoring or main_tpg_scoring
	order = {r.name or f'Round {r.number}': r.number for r in rounds}
	return Simulation(targets, order, pics, scoring, strategy, use_haversine=use_haversine)


def _add_submission_summary(row: dict[str, Any], sub: Submission, col_name: str, prefix: str = ''):
	row[col_name] = sub.name
	row[f'{prefix}score'] = sub.score
	row[f'{prefix}distance'] = sub.distance
	row[f'{prefix}lat'] = sub.latitude
	row[f'{prefix}lng'] = sub.longitude
	if sub.description:
		row[f'{prefix}description'] = sub.description


def get_round_summary(
	new_rounds: Iterable[Round] | Simulation,
	*,
	include_podium: bool = True,
	include_loser: bool = True,
) -> pandas.DataFrame:
	"""Returns a summary of who won (and optionally, who got podium for/who lost) each round. If new_rounds is an iterable of rounds, it is expected to be the output of Simulation.simulation_rounds(), and therefore have submissions scored and sorted by rank."""
	if isinstance(new_rounds, Simulation):
		new_rounds = new_rounds.simulate_rounds()

	rows = []
	for r in new_rounds:
		assert r.name is not None, 'why is r.name None'
		# sub.score/sub.distance should always be non-None but this is just to keep the type checker happy, unless we really want to go through the tedium of inventing a ScoredRound type just for that
		average_distance = mean(sub.distance for sub in r.submissions if sub.distance is not None)
		row = {
			'round': r.name,
			'average_score': mean(sub.score for sub in r.submissions if sub.score is not None),
			'average_distance': average_distance,
			'num_closer_than_average': sum(
				sub.distance is not None and sub.distance < average_distance
				for sub in r.submissions
			),
		}
		# Submissions of simulated rounds are already sorted
		winner = r.submissions[0]
		_add_submission_summary(row, winner, 'winner')
		if include_podium:
			_add_submission_summary(row, r.submissions[1], 'silver', 'silver_')
			_add_submission_summary(row, r.submissions[2], 'bronze', 'bronze_')
		if include_loser:
			_add_submission_summary(row, r.submissions[-1], 'loser', 'loser_')
		rows.append(row)
	return pandas.DataFrame(rows).set_index('round')


def get_player_summary(new_rounds: Iterable[Round] | Simulation) -> pandas.DataFrame:
	"""Returns a leaderboard-like summary of each simulated player's results and best/worst rounds from a simulation. If new_rounds is an iterable of rounds, it is expected to be the result of Simulation.simulate_rounds(), and therefore have names for each round and scores/distances for each submission."""
	if isinstance(new_rounds, Simulation):
		new_rounds = new_rounds.simulate_rounds()

	scores_by_round: defaultdict[str, dict[str, float]] = defaultdict(dict)
	total_distances: dict[str, float] = {}
	times_above_average: dict[str, int] = {}
	ranks_by_round: defaultdict[str, list[int]] = defaultdict(list)
	for r in new_rounds:
		assert r.name is not None, 'why is r.name None'
		average_distance = mean(sub.distance for sub in r.submissions if sub.distance is not None)

		for sub in r.submissions:
			assert sub.score is not None, 'why is sub.score None'
			assert sub.distance is not None, 'why is sub.distance None'
			assert sub.rank is not None, 'why is sub.rank None'
			scores_by_round[sub.name][r.name] = sub.score
			ranks_by_round[sub.name].append(sub.rank)

			if sub.name not in total_distances:
				total_distances[sub.name] = 0
			total_distances[sub.name] += sub.distance
			if sub.name not in times_above_average:
				times_above_average[sub.name] = 0
			times_above_average[sub.name] += sub.distance < average_distance

	rows = []
	valuegetter = itemgetter(1)
	for name, scores in scores_by_round.items():
		row = {'name': name, 'total': sum(scores.values())}
		row['best_round'], row['best_score'] = max(scores.items(), key=valuegetter)
		row['worst_round'], row['worst_score'] = min(scores.items(), key=valuegetter)
		row['total_distance'] = total_distances[name]
		row['times_closer_than_average'] = times_above_average[name]
		
		rank_counter = Counter(ranks_by_round[name])
		row['rounds_won'] = rank_counter[1]
		row['rounds_podiummed'] = rank_counter[1] + rank_counter[2] + rank_counter[3]
		row['rounds_lost'] = rank_counter[len(scores_by_round)]
		
		rows.append(row)
	return (
		pandas.DataFrame(rows)
		.set_index('name', verify_integrity=True)
		.sort_values('total', ascending=False)
	)
