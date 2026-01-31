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


def _set_index_name_col[T: 'pandas.DataFrame'](
	df: T, name_col: Hashable, log_context: Any = None, *, log_not_contains: bool = True
) -> T | None:
	"""Returns df with the new index if we successfully set the index, or None if we did not."""
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
		return df.set_index(name_col)
	except ValueError as ex:
		logger.info('Tried to set index of %s to %s but got error: %s', log_context, name_col, ex)
		return None


def _maybe_set_index_inner[T: 'pandas.DataFrame'](
	df: T,
	cols_to_try: Iterable[Hashable],
	log_context: Any = None,
	*,
	log_not_contains: bool = True,
) -> tuple[T | None, Hashable | None]:
	for col_name in cols_to_try:
		new_df = _set_index_name_col(df, col_name, log_context, log_not_contains=log_not_contains)
		if new_df is not None:
			return new_df, col_name
	return None, None


maybe_name_cols = ('name', 'Name', 'desc', 'description', 'Name')
"""Column names which are maybe the name of something."""


def first_unique_column_label(df: 'pandas.DataFrame') -> Hashable | None:
	"""Finds the first column label in `df` that is not a geometry column and has unique values."""
	return next(
		(name for name, col in df.items() if col.dtype != 'geometry' and col.is_unique), None
	)


def _maybe_set_name_from_pattern[T: 'pandas.DataFrame'](
	df: T, log_context: Any = None
) -> tuple[T | None, Hashable | None]:
	cols_lower = df.columns.str.lower()
	ends_with_name = cols_lower[cols_lower.str.endswith('_name_')]
	if not ends_with_name.empty:
		new_df, col_name = _maybe_set_index_inner(df, ends_with_name, log_context)
		if new_df is not None:
			return new_df, col_name
	contains_name = cols_lower[cols_lower.str.contains('_name_', regex=False)]
	if not contains_name.empty:
		new_df, col_name = _maybe_set_index_inner(df, contains_name, log_context)
		if new_df is not None:
			return new_df, col_name

	return None, None


def maybe_set_index_name_col[T: 'pandas.DataFrame'](
	df: T,
	name_col: Hashable | None = None,
	log_context: Any = None,
	*,
	try_autodetect: bool = True,
	try_pattern_match: bool = True,
	try_first_unique: bool = False,
) -> tuple[T, Hashable | None]:
	"""Attempts to set the index of a DataFrame and return the new copy, if and only if it is unique. Can optionally try and detect columns that look like they contain names, or fall back to the first unique column.

	Returns:
		tuple: (DataFrame which may be a new copy with the new index, name column if it was set succesfully or None if not)
	"""
	if name_col:
		new_df = _set_index_name_col(df, name_col, log_context)
		if new_df is not None:
			return new_df, name_col

	if try_autodetect:
		new_df, col_name = _maybe_set_index_inner(
			df, maybe_name_cols, log_context, log_not_contains=False
		)
		if new_df is not None:
			return new_df, col_name
	if try_pattern_match:
		new_df, col_name = _maybe_set_name_from_pattern(df, log_context)
		if new_df is not None:
			return new_df, col_name

	if try_first_unique:
		first_unique = first_unique_column_label(df)
		if first_unique:
			return df.set_index(first_unique), first_unique

	return df, None


def try_auto_set_index[T: 'pandas.DataFrame'](
	df: T,
	col_name: Hashable | None = None,
	log_context: Any | None = None,
	*,
	try_first_unique: bool = False,
) -> T:
	"""Attempts to set the index of `df` to something that looks like a name column, or a user-provided name column (`try_first`), but only if that column is unique."""
	df, _new_name_col = maybe_set_index_name_col(
		df,
		col_name,
		log_context,
		try_autodetect=True,
		try_pattern_match=True,
		try_first_unique=try_first_unique,
	)
	return df


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
