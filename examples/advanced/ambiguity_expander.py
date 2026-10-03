"""
Expanding Ambiguous Parses
==========================

When the Earley parser runs with ``ambiguity='explicit'``, ambiguous
derivations come back wrapped in ``_ambig`` nodes (a parse forest).
``AmbiguityExpander`` iterates over the individual parse trees lazily -
one at a time, in a fixed order - instead of materializing (or silently
dropping) the combinations.

This demo parses an ambiguous expression, prints how many parses exist,
and shows the structure of each one.
"""

from lark import Lark
from lark.ambiguity import AmbiguityExpander

grammar = r"""
    start: e
    e: e PLUS e
     | NUM
    PLUS: "+"
    NUM: /[0-9]+/
    %ignore " "
"""

parser = Lark(grammar, parser='earley', ambiguity='explicit')
forest = parser.parse('1+2+3')

expander = AmbiguityExpander(forest)
print('The forest has', expander.total, 'parse(s):')
for i, tree in enumerate(expander, 1):
    print('--- parse %d ---' % i)
    print(tree.pretty())

# A cap can be set when the forest may be huge; truncation is reported
# instead of the first parse being silently taken as the only one.
capped = AmbiguityExpander(forest, max_expansions=1)
print('With max_expansions=1: got', len(list(capped)),
      'parse(s), truncated =', capped.truncated)
