"""
Expanding Ambiguities Lazily
=============================

A short demonstration of :class:`lark.visitors.ExpandAmbiguities`.

Parsing with ``ambiguity='explicit'`` produces a single tree whose ``_ambig``
nodes mark where the grammar was ambiguous. ``ExpandAmbiguities`` yields each
unambiguous tree one at a time, instead of materializing the whole Cartesian
product up-front.

This example shows how to:

* iterate the unambiguous trees lazily,
* cap the number of results and check the ``truncated`` status,
* print the count and structure of each result.
"""
from lark import Lark
from lark.visitors import ExpandAmbiguities


grammar = """
    !start: x y

    !x: "a" "b"
      | "ab"
      | "abc"

    !y: "c" "d"
      | "cd"
      | "d"
"""

parser = Lark(grammar, ambiguity='explicit')
tree = parser.parse('abcd')

# Yield at most 3 trees; report whether more exist via `truncated`.
expander = ExpandAmbiguities(tree, max_trees=3)
for i, result in enumerate(expander):
    print(f"--- result {i} ---")
    print(result.pretty())

print(f"yielded {expander.count} tree(s), truncated={expander.truncated}")

# Each result is independent: mutating one never affects the others.
results = list(ExpandAmbiguities(tree))
results[0].data = 'mutated'
print(f"first result mutated; second result still: {results[1].data}")
