"""Explicit reduce-coverage collection for the LALR(1) parser.

This module implements an opt-in coverage collector that records, during the
actual LALR reduce path, which rules (and which aliased alternatives) were
reduced while parsing a single input.

Unlike inspecting the resulting parse tree, this observes the parser itself:

- Hidden rules (``_rule``) and inlined rules (``?rule``) still reduce, but may
  not appear as nodes in the output tree.
- Aliases rename tree nodes, so the tree alone cannot tell which alternative
  of a rule was used.

Enable it with ``Lark(..., parser='lalr', coverage=True)`` and read the
per-parse report from ``Lark.coverage_report``, or use
``Lark.parse_with_coverage()`` which returns ``(result, report)``.

Supported scope:

- Only ``parser='lalr'`` (any lexer it supports). Earley/CYK are not supported.
- Collection happens in ``Lark.parse()`` / ``Lark.parse_with_coverage()``.
  ``parse_interactive()`` and ``scan()`` do not collect coverage.
- Only reductions that actually completed are recorded. If parsing fails,
  the report keeps the reductions completed so far and is marked with
  ``success == False``. Speculative/predicted branches are never counted.
- Each call to ``parse()`` produces a fresh, independent report. Counts are
  never accumulated across parses of the same ``Lark`` instance.
- User transformers/lexer callbacks are not wrapped or affected; the returned
  tree and the number of callback invocations are identical to a parse
  without coverage.
"""

###{standalone
from collections import Counter


class CoverageReport:
    """Records which rules (and alternatives) were reduced during a single parse.

    A fresh instance is created for every ``Lark.parse()`` call when the
    ``coverage`` option is enabled.

    Attributes:
        counts: A ``Counter`` mapping ``(rule_name, alias)`` to the number of
            times that exact alternative was reduced. ``alias`` is ``None``
            for alternatives without an alias.
        reductions: The ordered sequence of ``(rule_name, alias)`` reductions,
            in the order the parser performed them.
        success: ``None`` while the parse is still in progress, ``True`` if
            the parse completed successfully, ``False`` if it raised
            ``UnexpectedInput`` (the reductions completed before the failure
            are kept).
    """

    def __init__(self):
        self.counts = Counter()
        self.reductions = []
        self.success = None

    def record(self, rule):
        """Record one completed reduction of ``rule`` (called by the parser)."""
        key = (rule.origin.name, rule.alias)
        self.counts[key] += 1
        self.reductions.append(key)

    @property
    def total(self):
        """Total number of completed reductions."""
        return sum(self.counts.values())

    def rule_count(self, rule_name, alias=None):
        """Number of times the given alternative was reduced (0 if never)."""
        return self.counts.get((rule_name, alias), 0)

    def aliases(self, rule_name):
        """Mapping of ``alias -> count`` for every recorded alternative of a rule."""
        return {alias: count for (name, alias), count in self.counts.items()
                if name == rule_name}

    def rules(self):
        """Names of all rules that were reduced at least once."""
        return sorted({name for name, _alias in self.counts})

    def pretty(self):
        """A short human-readable summary of the report."""
        status = {None: 'in-progress', True: 'success', False: 'failed'}[self.success]
        lines = ['CoverageReport(%s, %d reductions)' % (status, self.total)]
        for (name, alias), count in sorted(self.counts.items(), key=lambda item: (item[0][0], item[0][1] or '')):
            label = '%s -> %s' % (name, alias) if alias is not None else name
            lines.append('  %-30s %d' % (label, count))
        return '\n'.join(lines)

    def __str__(self):
        return self.pretty()

    def __repr__(self):
        return 'CoverageReport(success=%r, counts=%r)' % (self.success, dict(self.counts))
###}
