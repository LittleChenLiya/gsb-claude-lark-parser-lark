"""LALR Reduce-Coverage Report

Demonstrates the explicit coverage collector (``coverage=True``), which
records - on the parser's real reduce path - which rules and which aliased
alternatives were actually reduced during a parse.

This sees what the output tree cannot: inline/hidden rules leave no tree
nodes, and aliases rename them, so tree traversal cannot reconstruct which
alternatives fired.

Supported scope: ``parser='lalr'`` only. Each ``parse()`` call produces a
fresh, independent report; a failed parse keeps the reductions completed so
far and marks the report with ``success == False``.
"""
from lark import Lark, UnexpectedInput

grammar = r"""
    ?start: sum
    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub
    ?product: NUMBER
    NUMBER: /\d+/
    %ignore /\s+/
"""

parser = Lark(grammar, parser='lalr', coverage=True)

for text in ("1+2", "10-3-2"):
    tree, report = parser.parse_with_coverage(text)
    print("Parse %r ->" % text, tree.pretty(), sep='\n')
    print(report.pretty())
    print()

# A failed parse keeps the reductions that completed before the failure,
# without counting branches that were predicted but never reduced.
try:
    parser.parse("1+")
except UnexpectedInput as e:
    print("Parse '1+' failed:", type(e).__name__)
    print(parser.coverage_report.pretty())
