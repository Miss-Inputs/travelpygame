from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas


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


def custom_tpg_score(
	distances: 'pandas.Series',
	world_distance: float = 20_000.0,
	fivek_score: float | None = 7_500.0,
	fivek_threshold: float | None = 0.1,
	*,
	allow_negative: bool = False,
):
	"""
	Computes the score for a whole round of TPG, with a custom world distance constant, for spinoff TPGs that cover a smaller area. Does not factor in ties for now. Rounds to 2 decimal places as normal.

	Arguments:
		distances: Distances in kilometres for each round.
		world_distance: Maximum distance possible in this subset of the world in kilometres, defaults to 20K which is the default constant (not the exact max distance of the earth but close enough) anyway.
		fivek_score: Flat score for 5Ks, or None to disable this / consider 5Ks manually.
		fivek_threshold: 5K threshold in km, defaults to 100m
		allow_negative: Allow distance scores to be negative, if false (default) give a score of 0 if distance is greater than world_distance
	"""
	distance_scores = world_distance - distances
	if not allow_negative:
		distance_scores = distance_scores.clip(0)
	players_beaten = distances.size - distances.rank(method='max', ascending=True)
	players_beaten_scores = 5000 * (players_beaten / (distances.size - 1))
	scores = (distance_scores + players_beaten_scores) / 2
	if fivek_score:
		scores[distances <= fivek_threshold] = fivek_score
	return scores.round(2)

