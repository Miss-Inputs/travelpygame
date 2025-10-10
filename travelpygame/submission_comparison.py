"""Functions for getting comparisons of submissions in a round, to see closest differences, etc."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy

from .util import get_distances

if TYPE_CHECKING:
	from shapely import Point

	from .tpg_data import Round, Submission

logger = logging.getLogger(__name__)


@dataclass
class SubmissionDifference:
	round_name: str | None
	"""Name of round that we are comparing."""
	target: 'Point'
	"""Target for the round that we are comparing."""
	round_num_players: int
	"""Total number of submissions for this round, for the benefit of comparison."""

	player: str
	"""Name of the player who we are following the perspective of."""
	player_pic: 'Point'
	"""The location of the player's submission."""
	player_score: float | None
	"""The player's score in this round."""
	player_distance: float
	"""The player's distance to the target."""
	player_placing: int
	"""The player's ranking out of all submissions for that round."""

	rival: str
	"""Name of player who we are comparing to, because they are one spot above us, etc."""
	rival_pic: 'Point'
	"""The location of the rival's submission."""
	rival_score: float | None
	"""The rival's score in this round."""
	rival_distance: float
	"""The rival's distance to the target."""
	# rival_placing would in theory just be player_placing + 1

	@property
	def score_diff(self) -> float | None:
		if self.player_score is None or self.rival_score is None:
			return None
		return self.rival_score - self.player_score

	@property
	def distance_diff(self) -> float:
		return self.player_distance - self.rival_distance


def find_all_next_highest_placings(
	round_: 'Round', *, by_score: bool = False, use_haversine: bool = True
) -> Iterator[SubmissionDifference]:
	if not round_.is_scored:
		if by_score:
			raise ValueError('Round is not scored, so you will want to do that yourself')
		points = numpy.asarray([(sub.longitude, sub.latitude) for sub in round_.submissions])
		a = get_distances((round_.latitude, round_.longitude), points, use_haversine=use_haversine)
		subs_and_indices = sorted(enumerate(round_.submissions), key=lambda i_sub: a[i_sub[0]])
		sorted_subs = [sub for _, sub in subs_and_indices]
		# Doing it this way will calculate the distance twice, oh well, don't feel like refactoring right now
	else:
		sorted_subs = (
			sorted(round_.submissions, key=attrgetter('score'), reverse=True)
			if by_score
			else sorted(round_.submissions, key=attrgetter('distance'))
		)
	for i, (rival, player) in enumerate(pairwise(sorted_subs), 2):
		assert player.distance is not None, 'player.distance is None'
		assert rival.distance is not None, 'rival.distance is None'
		yield SubmissionDifference(
			round_.name,
			round_.target,
			len(sorted_subs),
			player.name,
			player.point,
			player.score,
			player.distance,
			i,
			rival.name,
			rival.point,
			rival.score,
			rival.distance,
		)


def find_next_highest_placing(
	round_: 'Round', submission: 'Submission', *, by_score: bool = False, use_haversine: bool = True
) -> SubmissionDifference | None:
	if not round_.is_scored:
		if by_score:
			raise ValueError('Round is not scored, so you will want to do that yourself')
		points = numpy.asarray([(sub.longitude, sub.latitude) for sub in round_.submissions])
		a = get_distances((round_.latitude, round_.longitude), points, use_haversine=use_haversine)
		for i in range(len(round_.submissions)):
			round_.submissions[i].distance = a[i]
		sorted_subs = sorted(round_.submissions, key=attrgetter('distance'))
	else:
		sorted_subs = (
			sorted(round_.submissions, key=attrgetter('score'), reverse=True)
			if by_score
			else sorted(round_.submissions, key=attrgetter('distance'))
		)
	index = sorted_subs.index(submission)
	if index == 0:
		# You won! That's allowed
		return None
	next_highest = sorted_subs[index - 1]
	assert submission.distance is not None, 'submission.distance is None'
	assert next_highest.distance is not None, 'next_highest.distance is None'
	return SubmissionDifference(
		round_.name,
		round_.target,
		len(sorted_subs),
		submission.name,
		submission.point,
		submission.score,
		submission.distance,
		index + 1,
		next_highest.name,
		next_highest.point,
		next_highest.score,
		next_highest.distance,
	)


def compare_player_in_round(
	round_: 'Round', name: str, *, by_score: bool = False, use_haversine: bool = True
) -> SubmissionDifference | None:
	player_submission = round_.find_player(name)
	if not player_submission:
		# We did not submit for this round, and that's okay
		return None
	return find_next_highest_placing(
		round_, player_submission, use_haversine=use_haversine, by_score=by_score
	)


def find_all_closest_placings(
	rounds: list['Round'], name: str | None, *, by_score: bool = False, use_haversine: bool = True
) -> Iterator[SubmissionDifference]:
	for round_ in rounds:
		if name:
			player_submission = round_.find_player(name)
			if not player_submission:
				# We did not submit for this round, and that's okay
				continue
			diff = find_next_highest_placing(
				round_, player_submission, use_haversine=use_haversine, by_score=by_score
			)
			if diff:
				yield diff
		else:
			yield from find_all_next_highest_placings(
				round_, by_score=by_score, use_haversine=use_haversine
			)
