"""Functions to deal with graphs (in the mathematical sense of the word, not the office/spreadsheet sense of the word)."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pandas import DataFrame


def to_graph(
	df: 'DataFrame',
	source_col: str | None,
	dest_col: str,
	weight_col: str | None,
	output_path: Path,
):
	"""Outputs a dot file containing a directed graph from a DataFrame, where each row contains the start node for each edge (in either a specific column, or the index) and the end node in another column.

	Will write text as UTF-8 and uses synchronous I/O.

	Arguments:
		df: pandas DataFrame.
		source_col: The name of the column containing the name of source nodes. If None, uses the index.
		dest_col: The name of the column containing the name of destination nodes.
		weight_col: Optionally, the name of a column containing weights to be added as attributes to each edge. The weight attribute is always called "weight", regardless of what this column is called.
		output_path: Path object specifying a path to be written to (does not check if this already exists or anything like that).
	"""
	with output_path.open('wt', encoding='utf8') as f:
		f.write('digraph "" {\n')
		for index, row in df.iterrows():
			source = str(index if source_col is None else row[source_col])
			source = source.replace('"', '\\"')
			dest = str(row[dest_col])
			dest = dest.replace('"', '\\"')
			line = f'"{source}" -> "{dest}"'
			if weight_col:
				line += f' [weight={row[weight_col]}]'
			f.write(f'{line};\n')
		f.write('}')
