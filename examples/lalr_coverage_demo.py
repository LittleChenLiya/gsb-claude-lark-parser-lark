"""Demo: LALR coverage collection.

This example shows how to use the coverage collector to see which rules and
alternatives were actually reduced during LALR parsing. This is useful for
testing grammars and understanding parse behavior.

The coverage collector records reductions at the actual reduce path, which
means it captures the real derivation steps including inline rules and aliases
that would be hidden if you only traversed the result tree.
"""

import os
import sys

# Ensure the worktree's lark package is imported, not the site-packages version
_worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _worktree_root not in sys.path:
    sys.path.insert(0, _worktree_root)

from lark import Lark, Transformer
from lark.exceptions import UnexpectedInput
from lark.parsers.lalr_coverage import CoverageCollector


def main():
    # Grammar with two alias branches and one inline rule
    grammar = """
    start: branch_a | branch_b | _inline
    branch_a: "a" -> a
    branch_b: "b" -> b
    _inline: "x"
    """

    lark = Lark(grammar, parser='lalr')

    print("=" * 60)
    print("LALR Coverage Collection Demo")
    print("=" * 60)

    # Parse different inputs and show coverage
    for text in ['a', 'b', 'x']:
        tree, collector = lark.parse(text, coverage=True)
        print(f"\nInput: {text!r}")
        print(f"  Tree: {tree}")
        print(f"  {collector.report()}")

    # Show that tree is identical with and without coverage
    print("\n" + "=" * 60)
    print("Tree consistency check")
    print("=" * 60)
    tree_default = lark.parse('a')
    tree_coverage, _ = lark.parse('a', coverage=True)
    print(f"  Tree identical with/without coverage: {tree_default == tree_coverage}")

    # Show parse failure keeps completed reductions
    print("\n" + "=" * 60)
    print("Parse failure handling")
    print("=" * 60)
    grammar2 = """
    start: a b c | d
    a: "a"
    b: "b"
    c: "c"
    d: "d"
    """
    lark2 = Lark(grammar2, parser='lalr', lexer='basic')
    try:
        lark2.parse('abcd', coverage=True)
    except UnexpectedInput as e:
        if hasattr(e, 'coverage'):
            print(f"  Parse failed as expected")
            print(f"  {e.coverage.report()}")

    # Show Transformer call count is unchanged
    print("\n" + "=" * 60)
    print("Transformer call count check")
    print("=" * 60)

    class CountingTransformer(Transformer):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def a(self, children):
            self.calls += 1
            return 'a_called'

    t1 = CountingTransformer()
    lark1 = Lark(grammar, parser='lalr', transformer=t1)
    result1 = lark1.parse('a')

    t2 = CountingTransformer()
    lark2 = Lark(grammar, parser='lalr', transformer=t2)
    result2, _ = lark2.parse('a', coverage=True)

    print(f"  Calls without coverage: {t1.calls}")
    print(f"  Calls with coverage: {t2.calls}")
    print(f"  Call count unchanged: {t1.calls == t2.calls}")
    print(f"  Result unchanged: {result1 == result2}")

    print("\n" + "=" * 60)
    print("Support scope")
    print("=" * 60)
    print("  - Parser: LALR only")
    print("  - Lexer: all lexers supported by LALR (basic, contextual)")
    print("  - Features: aliases, inline rules, transformers")
    print("  - Default behavior: unchanged (no coverage unless requested)")
    print("  - Independence: each parse has its own collector")


if __name__ == '__main__':
    main()
