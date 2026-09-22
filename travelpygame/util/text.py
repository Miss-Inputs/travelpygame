"""Some string manipulation functions, just because"""
import re
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

_emoji_regex_parts = (
	# This comment is so it lines up nicely when autoformatted
	r'\uE000-\uF8FF',
	r'\U0001F100-\U0001F2FF',
	r'\U0001F300-\U0001F6FF',
	r'\U0001F780-\U0001F9FF',
	r'\U0001FA70-\U0001FAFF',
	r'\U000E0000-\U000E007F',
)
_emoji_regex_middle = ''.join(_emoji_regex_parts)
probably_emoji = re.compile(rf'[{_emoji_regex_middle}]+')
"""Emoji regex kinda, excludes code blocks:
	E000 Private Use Area
	1F100 Enclosed Alphanumeric Supplement
	1F200 Enclosed Ideographic Supplement
	1F300 Misc Symbols and Pictographs
	1F600 Emoticons
	1F650 Ornamental Dingbats
	1F680 Transport and Map Symbols
	1F780 Geometric Shapes Extended
	1F900 Supplemental Symbols and Pictographs
	1FA70 Symbols and Pictographs Extended-A
	E0000 Tags
Unashamedly has false positives and false negatives, as this will never realistically be perfect unless I import a library just to do this which seems a bit much.
"""


def normalize_player_name(name: str):
	name = probably_emoji.sub('', name)
	# Probably more that can be done there, but that will do the trick
	return name.strip(' ;:|§.')
