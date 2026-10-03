"""Tests for the explicit LALR reduce-coverage collector (coverage=True)."""

from unittest import TestCase, main

from lark import Lark, CoverageReport, Tree
from lark.exceptions import ConfigurationError, UnexpectedInput
from lark.visitors import Transformer


# Two aliased alternatives (add / sub) plus inline rules (?sum, ?product)
# whose nodes do not necessarily appear in the output tree.
GRAMMAR = r"""
?start: sum
?sum: product
    | sum "+" product   -> add
    | sum "-" product   -> sub
?product: NUMBER
NUMBER: /\d+/
%ignore /\s+/
"""


class CountingTransformer(Transformer):
    """Records how many times each user callback fires."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def add(self, children):
        self.calls.append('add')
        return ('add', children)

    def sub(self, children):
        self.calls.append('sub')
        return ('sub', children)

    def NUMBER(self, token):
        self.calls.append('NUMBER')
        return int(token)


class TestCoverage(TestCase):
    def make_parser(self, **kwargs):
        kwargs.setdefault('parser', 'lalr')
        return Lark(GRAMMAR, coverage=True, **kwargs)

    def test_aliased_branch_counts(self):
        parser = self.make_parser()
        parser.parse("1+2")
        report = parser.coverage_report
        self.assertIsInstance(report, CoverageReport)
        self.assertTrue(report.success)
        # The real alternative identity is recorded on the reduce path
        self.assertEqual(report.rule_count('sum', 'add'), 1)
        self.assertEqual(report.rule_count('sum', 'sub'), 0)

        parser.parse("4-5-6")
        report = parser.coverage_report
        self.assertEqual(report.rule_count('sum', 'sub'), 2)
        self.assertEqual(report.rule_count('sum', 'add'), 0)

        parser.parse("1+2-3")
        report = parser.coverage_report
        self.assertEqual(report.rule_count('sum', 'add'), 1)
        self.assertEqual(report.rule_count('sum', 'sub'), 1)

    def test_inline_rules_are_recorded(self):
        # ?sum / ?product nodes are inlined away in the tree, but the
        # reductions really happen and must be visible in the report.
        parser = self.make_parser()
        tree = parser.parse("1+2")
        self.assertEqual(tree.data, 'add')  # no 'sum'/'product' nodes in the tree
        report = parser.coverage_report
        self.assertEqual(report.rule_count('product'), 2)
        self.assertEqual(report.rule_count('sum'), 1)       # ?sum: product
        self.assertEqual(report.rule_count('sum', 'add'), 1)
        self.assertEqual(report.rule_count('start'), 1)
        self.assertEqual(report.total, 5)
        self.assertIn(('sum', 'add'), report.reductions)

    def test_tree_identical_without_coverage(self):
        covered = self.make_parser()
        plain = Lark(GRAMMAR, parser='lalr')
        for text in ("1", "1+2", "9-4+3-2"):
            self.assertEqual(covered.parse(text), plain.parse(text))
        self.assertIsNone(plain.coverage_report)  # default parsing unchanged

    def test_failed_parse_keeps_completed_reductions(self):
        parser = self.make_parser()
        parser.parse("1+2")  # previous successful report must be replaced
        with self.assertRaises(UnexpectedInput):
            parser.parse("3+")
        report = parser.coverage_report
        self.assertIsNotNone(report)
        self.assertFalse(report.success)
        # Completed reductions are kept...
        self.assertEqual(report.rule_count('product'), 1)
        self.assertEqual(report.rule_count('sum'), 1)
        # ...but the predicted 'add' branch was never reduced, so not counted
        self.assertEqual(report.rule_count('sum', 'add'), 0)
        self.assertEqual(report.total, 2)

    def test_reports_are_independent_per_parse(self):
        parser = self.make_parser()
        parser.parse("1+2+3")
        first = parser.coverage_report
        self.assertEqual(first.rule_count('sum', 'add'), 2)

        parser.parse("4-5")
        second = parser.coverage_report
        self.assertIsNot(first, second)
        # No counts leak from the first parse into the second
        self.assertEqual(second.rule_count('sum', 'add'), 0)
        self.assertEqual(second.rule_count('sum', 'sub'), 1)
        # The first report is untouched by the second parse
        self.assertEqual(first.rule_count('sum', 'add'), 2)
        self.assertEqual(first.total, sum(first.counts.values()))

    def test_repeated_parse_same_input_not_cumulative(self):
        parser = self.make_parser()
        parser.parse("1+2")
        r1 = parser.coverage_report
        parser.parse("1+2")
        r2 = parser.coverage_report
        self.assertIsNot(r1, r2)
        self.assertEqual(r1.counts, r2.counts)
        self.assertEqual(r2.total, r1.total)

    def test_transformer_calls_and_result_unchanged(self):
        t_plain = CountingTransformer()
        plain = Lark(GRAMMAR, parser='lalr', transformer=t_plain)

        t_covered = CountingTransformer()
        covered = Lark(GRAMMAR, parser='lalr', transformer=t_covered, coverage=True)

        for text in ("1+2", "7-8+9"):
            res_plain = plain.parse(text)
            res_covered, report = covered.parse_with_coverage(text)
            self.assertEqual(res_plain, res_covered)
            self.assertEqual(t_plain.calls, t_covered.calls)
            # Coverage still sees the reductions even with a transformer
            self.assertGreater(report.total, 0)
            t_plain.calls[:] = []
            t_covered.calls[:] = []

    def test_parse_with_coverage_returns_report(self):
        parser = self.make_parser()
        tree, report = parser.parse_with_coverage("1+2")
        self.assertIsInstance(tree, Tree)
        self.assertIs(report, parser.coverage_report)
        self.assertTrue(report.success)

    def test_parse_with_coverage_requires_option(self):
        parser = Lark(GRAMMAR, parser='lalr')
        with self.assertRaises(ConfigurationError):
            parser.parse_with_coverage("1+2")

    def test_coverage_requires_lalr(self):
        with self.assertRaises(ConfigurationError):
            Lark(GRAMMAR, coverage=True)  # defaults to parser='earley'
        with self.assertRaises(ConfigurationError):
            Lark(GRAMMAR, parser='cyk', coverage=True)

    def test_on_error_recovery_keeps_counting(self):
        parser = self.make_parser()
        errors = []

        def on_error(e):
            errors.append(e)
            return False

        with self.assertRaises(UnexpectedInput):
            parser.parse("1+2 $", on_error=on_error)
        report = parser.coverage_report
        self.assertFalse(report.success)
        # LALR reduces on lookahead: the bad token arrives before
        # 'sum: sum "+" product' is reduced, so it must not be counted.
        self.assertEqual(report.rule_count('sum', 'add'), 0)
        # '2' was shifted but never reduced, so only the prefix before it counts
        self.assertEqual(report.rule_count('product'), 1)
        self.assertEqual(report.rule_count('sum'), 1)
        self.assertEqual(len(errors), 1)

    def test_report_pretty_print(self):
        parser = self.make_parser()
        parser.parse("1+2")
        text = parser.coverage_report.pretty()
        self.assertIn('success', text)
        self.assertIn('sum -> add', text)
        self.assertIn('product', text)
        self.assertEqual(str(parser.coverage_report), text)


if __name__ == '__main__':
    main()
