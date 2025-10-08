"""Functions for getting comparisons of submissions in a round, to see closest differences, etc."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy
import shapely

from .util import geod_distance_and_bearing, get_distances, haversine_distance

if TYPE_CHECKING:
	from .tpg_data import Round, Submission

logger = logging.getLogger(__name__)


@dataclass
class SubmissionDifference:
	round_name: str | None
	"""Name of round that we are comparing."""
	target: shapely.Point
	"""Target for the round that we are comparing."""
	player: str
	"""Name of the player who we are following the perspective of."""
	player_pic: shapely.Point
	"""The location of the player's submission."""
	player_score: float | None
	"""The player's score in this round."""
	player_distance: float
	"""The player's distance to the target."""
	rival: str
	"""Name of player who we are comparing to, because they are one spot above us, etc."""
	rival_pic: shapely.Point
	"""The location of the rival's submission."""
	rival_score: float | None
	"""The rival's score in this round."""
	rival_distance: float
	"""The rival's distance to the target."""

	@property
	def score_diff(self) -> float | None:
		if self.player_score is None or self.rival_score is None:
			return None
		return self.rival_score - self.player_score

	@property
	def distance_diff(self) -> float:
		return self.player_distance - self.rival_distance

	@classmethod
	def compare(
		cls,
		round_: 'Round',
		player: 'Submission',
		rival: 'Submission',
		*,
		use_haversine: bool = True,
	):
		if player.distance is None:
			player_dist = (
				haversine_distance(
					round_.latitude, round_.longitude, player.latitude, player.longitude
				)
				if use_haversine
				else geod_distance_and_bearing(
					round_.latitude, round_.longitude, player.latitude, player.longitude
				)[0]
			)
		else:
			player_dist = player.distance
		if rival.distance is None:
			rival_dist = (
				haversine_distance(
					round_.latitude, round_.longitude, rival.latitude, rival.longitude
				)
				if use_haversine
				else geod_distance_and_bearing(
					round_.latitude, round_.longitude, rival.latitude, rival.longitude
				)[0]
			)
		else:
			rival_dist = rival.distance
		return cls(
			round_.name,
			shapely.Point(round_.longitude, round_.latitude),
			player.name,
			shapely.Point(player.longitude, player.latitude),
			player.score,
			player_dist,
			rival.name,
			shapely.Point(rival.longitude, rival.latitude),
			rival.score,
			rival_dist,
		)


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
	for rival, player in pairwise(sorted_subs):
		yield SubmissionDifference.compare(round_, player, rival, use_haversine=use_haversine)


def find_next_highest_placing(
	round_: 'Round', submission: 'Submission', *, by_score: bool = False, use_haversine: bool = True
) -> SubmissionDifference | None:
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
	index = sorted_subs.index(submission)
	if index == 0:
		# You won! That's allowed
		return None
	next_highest = sorted_subs[index - 1]
	return SubmissionDifference.compare(
		round_, submission, next_highest, use_haversine=use_haversine
	)


def compare_player_in_round(
	round_: 'Round', name: str, *, by_score: bool = False, use_haversine: bool = True
) -> SubmissionDifference | None:
	try:
		player_submission = next(sub for sub in round_.submissions if sub.name == name)
	except StopIteration:
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
			try:
				player_submission = next(sub for sub in round_.submissions if sub.name == name)
			except StopIteration:
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
