"""Coverage collection for LALR parsing.

This module provides a CoverageCollector class that records which rules and
alternatives were actually reduced during LALR parsing. This is useful for
testing grammars and understanding parse behavior.

The collector records reductions at the actual reduce path, not by traversing
the result tree. This is important because inline rules and aliases can hide
the actual reduction process.

Only successful reductions are counted. If a parse fails, the collector keeps
the completed reductions and marks the parse as failed.

Each parse should use its own collector instance to avoid mixing counts across
parses.
"""

from typing import Dict, Optional

from ..grammar import Rule


class CoverageCollector:
    """Collects coverage information during LALR parsing.

    Records which rules and alternatives were actually reduced during parsing.
    Each parse should use its own collector instance.

    The collector records reductions at the actual reduce path, which means it
    captures the real derivation steps including inline rules and aliases that
    would be hidden if you only traversed the result tree.

    Example:
        >>> collector = CoverageCollector()
        >>> tree = parser.parse(text, coverage=collector)
        >>> print(collector.report())
    """

    def __init__(self):
        self._rules: Dict[Rule, int] = {}
        self._aliases: Dict[str, int] = {}
        self._failed: bool = False

    def record_reduce(self, rule: Rule) -> None:
        """Record that a rule was successfully reduced.

        This is called at the actual reduce path in the LALR parser, so it
        captures the real derivation steps. Only successful reductions are
        recorded.

        Parameters:
            rule: The rule that was reduced.
        """
        self._rules[rule] = self._rules.get(rule, 0) + 1
        if rule.alias:
            self._aliases[rule.alias] = self._aliases.get(rule.alias, 0) + 1

    def mark_failed(self) -> None:
        """Mark the parse as failed.

        Called when a parse error occurs. The collector keeps the completed
        reductions, so you can see which rules were reduced before the failure.
        """
        self._failed = True

    @property
    def failed(self) -> bool:
        """Whether the parse failed.

        Returns:
            True if the parse failed, False otherwise.
        """
        return self._failed

    @property
    def rules(self) -> Dict[Rule, int]:
        """A copy of the rule counts.

        Returns:
            A dictionary mapping rules to their reduction counts.
        """
        return dict(self._rules)

    @property
    def aliases(self) -> Dict[str, int]:
        """A copy of the alias counts.

        Returns:
            A dictionary mapping aliases to their reduction counts.
        """
        return dict(self._aliases)

    @property
    def total_reductions(self) -> int:
        """The total number of successful reductions.

        Returns:
            The total count of rule reductions.
        """
        return sum(self._rules.values())

    def report(self) -> str:
        """Return a human-readable report of the coverage.

        Returns:
            A string describing the coverage information.
        """
        lines = []
        lines.append("Coverage Report:")
        lines.append(f"  Parse failed: {self._failed}")
        lines.append(f"  Total reductions: {self.total_reductions}")
        lines.append(f"  Unique rules reduced: {len(self._rules)}")
        if self._aliases:
            lines.append(f"  Unique aliases reduced: {len(self._aliases)}")
            lines.append("  Alias breakdown:")
            for alias, count in sorted(self._aliases.items()):
                lines.append(f"    {alias}: {count}")
        return '\n'.join(lines)

    def __repr__(self):
        return (f"<CoverageCollector total_reductions={self.total_reductions} "
                f"unique_rules={len(self._rules)} failed={self._failed}>")
