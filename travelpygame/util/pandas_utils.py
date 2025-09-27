import logging
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

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
