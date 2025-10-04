"""Functions to simulate TPG seasons, playing out rounds as though everyone was there to submit their best pic."""

import logging
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum, auto

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
	"""Simulated players will choose their closest pic. This is the sensible option."""
	Furthest = auto()
	"""Simulated players will choose their furthest away pic, for whatever reason."""
	Random = auto()


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
) -> list[Round]:
	pics = {
		player: shapely.points([(lng, lat) for lat, lng in latlngs]).tolist()
		for player, latlngs in get_submissions_per_user(rounds).items()
	}
	targets = {
		r.name or f'Round {r.number}': shapely.Point(r.longitude, r.latitude) for r in rounds
	}
	scoring = scoring or main_tpg_scoring
	order = {r.name or f'Round {r.number}': r.number for r in rounds}
	return Simulation(
		targets, order, pics, scoring, strategy, use_haversine=use_haversine
	).simulate_rounds()
