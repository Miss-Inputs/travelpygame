"""Some string manipulation functions, just because"""
from collections.abc import Sequence


def levenshtein_dist[T](s1: Sequence[T], s2: Sequence[T]) -> int:
	len1 = len(s1)
	len2 = len(s2)
	if len1 < len2:
		return levenshtein_dist(s2, s1)
	if not len2:
		return len1

	p = range(len2 + 1)
	for i, c1 in enumerate(s1):
		r = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = p[j + 1] + 1
			deletions = r[j] + 1
			subs = p[j] + (c1 != c2)
			r.append(min(insertions, deletions, subs))
		p = r
	return p[-1]
