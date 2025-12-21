
from enum import StrEnum

from pydantic import BaseModel, TypeAdapter
from shapely import Point

PlayerName = str
"""Type hint for what is a player name. Should be consistent."""


class Submission(BaseModel, extra='allow'):
	name: PlayerName
	"""Name of whoever submitted this."""
	latitude: float
	"""WGS84 latitude of the picture."""
	longitude: float
	"""WGS84 longitude of the picture."""
	description: str | None = None
	"""Some kind of description of the picture/location if we have one."""
	is_5k: bool | None = None
	"""Whether this submission counted as a 5K or not, or None if unknown."""
	is_antipode_5k: bool | None = None
	"""Whether this submission counted as a 5K for the antipode or not."""
	is_tie: bool = False
	"""Whether this submission should be considered to be a tie with other pics nearby that also have is_tie = True."""

	score: float | None = None
	"""Score for this submission, or None if score is not calculated yet."""
	rank: int | None = None
	"""Placement for this submission, starting at 1 for first place and 2 for second, etc, or None if score is not calculated yet."""
	distance: float | None = None
	"""Distance for this submission in metres from the target, or None if this is not calculated yet."""

	@property
	def point(self) -> Point:
		return Point(self.longitude, self.latitude)


class TPGType(StrEnum):
	Normal = 'normal'
	"""Standard TPG where you have one target per round and one submission."""
	# Anything else like multi (increasing) or line once I implement stuff like that


class RoundInfo(BaseModel, extra='allow'):
	"""I really am bad at naming things. This is Round but without the submissions"""

	name: str | None
	"""Round name, if applicable."""
	type: TPGType = TPGType.Normal
	"""For now…"""
	number: int
	season: int | None = None
	country_code: str | None = None
	"""Country code, if applicable/known beforehand."""
	latitude: float
	"""Latitude of round target."""
	longitude: float
	"""Longitude of round target."""

	@property
	def target(self) -> Point:
		return Point(self.longitude, self.latitude)


class Round(RoundInfo):
	"""A round + its submissions, which may or may not be scored."""

	submissions: list[Submission]

	@property
	def is_scored(self) -> bool:
		"""Returns true iff _all_ submissions are scored."""
		return all(sub.score is not None for sub in self.submissions)

	def find_player(self, name: str) -> Submission | None:
		"""Finds the submission of a player (case-sensitive, etc), or returns None if that player does not have a submission for this round."""
		return next((sub for sub in self.submissions if sub.name == name), None)


class ScoringOptions(BaseModel, extra='allow'):
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


class Season(BaseModel, extra='allow'):
	"""Stores a list of rounds and their submissions, and the rules for scoring. Not used just yet."""

	name: str
	"""Name of the spinoff, or "Main", etc"""
	number: int | None
	type: TPGType
	scoring: ScoringOptions
	"""Specifies how this season is scored."""
	rounds: list[Round]
	"""All rounds that have happened so far in this season."""

round_list_adapter = TypeAdapter(list[Round])
