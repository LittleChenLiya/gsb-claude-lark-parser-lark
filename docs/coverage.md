# Reduce Coverage (LALR)

Lark can collect an **explicit reduce-coverage report** while parsing: which
rules - and which aliased alternatives - were actually reduced by the parser.

This is recorded on the parser's real reduce path, not inferred from the
output tree. That matters because the tree hides information:

- Inline rules (`?rule`) and hidden rules (`_rule`) reduce, but may leave no
  node in the tree.
- Aliases (`-> name`) rename tree nodes, so the tree alone cannot say which
  alternative of a rule fired.

## Usage

```python
from lark import Lark

parser = Lark(grammar, parser='lalr', coverage=True)

tree, report = parser.parse_with_coverage("1+2")
print(report.pretty())
# CoverageReport(success, 5 reductions)
#   product                        2
#   start                          1
#   sum                            1
#   sum -> add                     1

# Or parse as usual and read the report afterwards:
tree = parser.parse("10-3")
report = parser.coverage_report
assert report.rule_count('sum', 'sub') == 1
```

See `examples/advanced/lalr_coverage.py` for a runnable demo.

## Semantics

- **Per-parse reports.** Every `parse()` call creates a fresh, independent
  `CoverageReport`. Counts are never accumulated across parses of the same
  `Lark` instance; `Lark.coverage_report` always refers to the most recent
  parse.
- **Only completed reductions count.** A reduction is recorded after its
  callback ran and the goto-shift succeeded. Branches the parser predicted
  but never reduced are not counted.
- **Failures keep the completed prefix.** If parsing raises
  `UnexpectedInput`, the report keeps every reduction completed before the
  failure and is marked with `report.success == False`.
- **No effect on results.** The returned tree (or transformer result) and
  the number of user transformer/lexer-callback invocations are identical
  to a parse without coverage. With `coverage=False` (the default), parsing
  is completely unchanged.

## Supported scope

- Only `parser='lalr'` (with any lexer it supports). Passing `coverage=True`
  with Earley or CYK raises `ConfigurationError`.
- Collection happens in `Lark.parse()` / `Lark.parse_with_coverage()`.
  `parse_interactive()` and `scan()` do not collect coverage.
- Works with `transformer`, `on_error`, `propagate_positions`, custom
  `tree_class`, and with parsers restored via `Lark.load()`.

## The CoverageReport object

- `counts` - a `Counter` of `(rule_name, alias) -> int` (`alias` is `None`
  for unaliased alternatives).
- `reductions` - the ordered sequence of `(rule_name, alias)` reductions.
- `success` - `True`/`False` once the parse finished, `None` while running.
- `total` - total number of completed reductions.
- `rule_count(rule, alias=None)` - count for one alternative.
- `aliases(rule)` - `{alias: count}` for all recorded alternatives of a rule.
- `rules()` - names of all rules that reduced at least once.
- `pretty()` - a short human-readable summary (also `str(report)`).
