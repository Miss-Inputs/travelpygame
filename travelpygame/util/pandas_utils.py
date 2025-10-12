import logging
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	import pandas

logger = logging.getLogger(__name__)


def find_first_matching_column(
	df: 'pandas.DataFrame', col_names: Iterable[Hashable]
) -> Hashable | None:
	for name in col_names:
		if name in df.columns:
			return name
	return None


maybe_name_cols = ('name', 'Name', 'desc', 'description', 'Name')
"""Column names which are maybe the name of something."""


def try_set_index_name_col[T: 'pandas.DataFrame'](df: T) -> T:
	name_col = find_first_matching_column(df, maybe_name_cols)
	if not name_col:
		return df
	# TODO: Try finding the first _unique_ column name
	try:
		return df.set_index(name_col, verify_integrity=True)
	except ValueError as ex:
		logger.info(ex)
		return df


def _maybe_set_index_name_col_inner[T: 'pandas.DataFrame'](
	df: T, name_col: Hashable, log_context: Any = None, *, log_not_contains: bool = True
) -> T | None:
	col = df.get(name_col)
	log_context = log_context or type(df).__name__
	if col is None:
		if log_not_contains:
			logger.info('%s did not contain %s', log_context, name_col)
		return None
	if not col.is_unique:
		logger.info('Column %s in %s is not unique', name_col, log_context)
		return None

	try:
		return df.set_index(name_col, verify_integrity=True)
	except ValueError as ex:
		logger.info('Tried to set index of %s to %s but got error: %s', log_context, name_col, ex)
		return None


def maybe_set_index_name_col[T: 'pandas.DataFrame'](
	df: T, name_col: Hashable | None, log_context: Any = None, *, try_autodetect: bool = True
) -> tuple[T, Hashable | None]:
	if name_col:
		new_df = _maybe_set_index_name_col_inner(df, name_col, log_context)
		if new_df is not None:
			return new_df, name_col

	if try_autodetect:
		for maybe_col_name in maybe_name_cols:
			new_df = _maybe_set_index_name_col_inner(
				df, maybe_col_name, log_context, log_not_contains=False
			)
			if new_df is not None:
				return new_df, maybe_col_name

	return df, None


def first_unique_column_label(df: 'pandas.DataFrame'):
	return next((name for name, col in df.items() if col.is_unique), None)


def detect_cat_cols(df: 'pandas.DataFrame', frac_threshold: int = 2):
	"""Quick and dirty way to detect which columns in a pandas DataFrame are probably categories, and therefore are useful groupings for stats etc. There are better ways to do this but if the user doesn't provide a list of category columns manually, this will do."""
	dtypes = df.dtypes
	cats = dtypes[dtypes == 'category'].index
	geom_cols = dtypes[dtypes == 'geometry'].index
	df = df.drop(columns=[*cats, *geom_cols])

	counts = df.count(axis='index')
	nunique = df[counts[counts > 1].index.to_list()].nunique()
	threshold = df.index.size // frac_threshold
	maybe_cats = nunique[(nunique > 1) & (nunique < threshold)]
	return [*cats, *maybe_cats.index]
