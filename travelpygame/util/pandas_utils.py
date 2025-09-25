from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas


def find_first_matching_column(df: 'pandas.DataFrame', col_names: Iterable[Hashable]) -> Hashable | None:
	for name in col_names:
		if name in df.columns:
			return name
	return None
