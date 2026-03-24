import re
from collections.abc import Sequence
from pathlib import Path

from travelpygame.util.kml import Placemark, SubmissionTracker, parse_submission_kml

from .classes import Round, Submission

bonus_points_regex = re.compile(r'\s*\(\+(\d+)\)$')


def _convert_submission_from_tracker(
	sub: Placemark, fivek_suffix: str | None = None, *, parse_bonus_points: bool
) -> Submission:
	extra = {}
	if sub.style:
		style_match = re.match(r'#(icon-.+?)-(.+?)-(.+)', sub.style)
		if style_match is None:
			extra['style'] = sub.style
		else:
			# Mind you, this is not the actual <Icon> tag in the thing, I'm not sure if that would be more useful
			extra['icon'] = style_match[1]
			extra['colour'] = style_match[2]
			extra['style'] = style_match[3]  # labelson, nodesc, etc

	is_5k = None
	name = sub.name
	if fivek_suffix is not None and name.endswith(fivek_suffix):
		name = name.removesuffix(fivek_suffix)
		is_5k = True

	bonus_points = None
	if parse_bonus_points:
		bonus_points_match = bonus_points_regex.search(name)
		if bonus_points_match is not None:
			name = name[: bonus_points_match.start()]
			bonus_points = int(bonus_points_match.group(1))

	return Submission(
		name=name,
		latitude=sub.point.y,
		longitude=sub.point.x,
		description=sub.description,
		is_5k=is_5k,
		bonus_points=bonus_points,
		**extra,  # pyright: ignore[reportArgumentType] #Pylance does not know how pydantic extra works, it seems
	)


def _find_season(season_starts: list[int], round_number: int) -> int:
	if 1 not in season_starts:
		season_starts.append(1)
	season_starts = sorted(season_starts, reverse=True)
	for season, season_start in enumerate(season_starts):
		if round_number >= season_start:
			return (len(season_starts) - season) - 1
	return 0


def convert_submission_tracker(
	tracker: Path | Sequence[Path] | SubmissionTracker,
	start_round: int = 1,
	season: int | list[int] | None = None,
	fivek_suffix: str | None = None,
	*,
	parse_bonus_points: bool = True,
) -> list[Round]:
	"""Converts a submission tracker (paths to .kml file(s), or a SubmissionTracker already parsed from such) to this more consistent format.

	Arguments:
		season: If a list, this is a list of starting rounds for each season (assumes there is a season zero, not because I can't be bothered handling one-based indexing, but because most spinoffs have a pilot).
		fivek_suffix: Optional string to put on the end of submission names to mark them manually as a 5K.
	"""
	if not isinstance(tracker, SubmissionTracker):
		tracker = parse_submission_kml(tracker)

	rounds: list[Round] = []
	for i, tracker_round in enumerate(tracker.rounds, start_round):
		round_season = _find_season(season, i) if isinstance(season, list) else season
		# TODO: Ensure that there are no duplicate names
		subs = [
			_convert_submission_from_tracker(
				sub, fivek_suffix, parse_bonus_points=parse_bonus_points
			)
			for sub in tracker_round.submissions
		]
		rounds.append(
			Round(
				name=tracker_round.name,
				number=i,
				season=round_season,
				latitude=tracker_round.target.y,
				longitude=tracker_round.target.x,
				submissions=subs,
			)
		)
	return rounds
