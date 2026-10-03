# -*- coding: utf-8 -*-
"""Tests for LALR coverage collection."""

import unittest

from lark import Lark, Transformer, Tree
from lark.exceptions import UnexpectedInput
from lark.parsers.lalr_coverage import CoverageCollector


class TestLALRCoverage(unittest.TestCase):
    """Tests for LALR coverage collection."""

    def setUp(self):
        self.grammar = """
        start: branch_a | branch_b
        branch_a: "a" -> a
        branch_b: "b" -> b
        _inline: "x"
        """
        self.lark = Lark(self.grammar, parser='lalr')

    def test_default_parsing_unchanged(self):
        tree = self.lark.parse('a')
        self.assertIsInstance(tree, Tree)
        self.assertEqual(tree.data, 'start')

    def test_coverage_returns_tuple(self):
        result = self.lark.parse('a', coverage=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        tree, collector = result
        self.assertIsInstance(tree, Tree)
        self.assertIsInstance(collector, CoverageCollector)

    def test_tree_identical_with_and_without_coverage(self):
        tree_default = self.lark.parse('a')
        tree_coverage, _ = self.lark.parse('a', coverage=True)
        self.assertEqual(tree_default, tree_coverage)

    def test_different_inputs_give_different_branch_counts(self):
        _, col_a = self.lark.parse('a', coverage=True)
        _, col_b = self.lark.parse('b', coverage=True)
        self.assertIn('a', col_a.aliases)
        self.assertNotIn('b', col_a.aliases)
        self.assertIn('b', col_b.aliases)
        self.assertNotIn('a', col_b.aliases)

    def test_inline_rule_recorded(self):
        grammar = """
        start: _inline
        _inline: "x"
        """
        lark = Lark(grammar, parser='lalr')
        _, col = lark.parse('x', coverage=True)
        self.assertGreater(col.total_reductions, 0)
        rule_names = [r.origin.name for r in col.rules]
        self.assertIn('_inline', rule_names)

    def test_parse_failure_keeps_completed_reductions(self):
        grammar = """
        start: "a" expr
        expr: "b" -> b | "c" -> c
        """
        lark = Lark(grammar, parser='lalr')
        try:
            lark.parse('abd', coverage=True)
            self.fail("Should have raised UnexpectedInput")
        except UnexpectedInput as e:
            self.assertTrue(hasattr(e, 'coverage'))
            col = e.coverage
            self.assertIsInstance(col, CoverageCollector)
            self.assertTrue(col.failed)

    def test_illegal_suffix_does_not_count_predicted_branches(self):
        grammar = """
        start: branch_a | branch_b
        branch_a: "a" -> a
        branch_b: "b" -> b
        """
        lark = Lark(grammar, parser='lalr')
        try:
            lark.parse('ac', coverage=True)
            self.fail("Should have raised UnexpectedInput")
        except UnexpectedInput as e:
            col = e.coverage
            self.assertTrue(col.failed)
            self.assertEqual(col.total_reductions, 0)

    def test_repeated_parse_independence(self):
        _, col1 = self.lark.parse('a', coverage=True)
        _, col2 = self.lark.parse('b', coverage=True)
        self.assertIsNot(col1, col2)
        self.assertIn('a', col1.aliases)
        self.assertNotIn('b', col1.aliases)
        self.assertIn('b', col2.aliases)
        self.assertNotIn('a', col2.aliases)

    def test_callback_count_unchanged(self):
        class CountingTransformer(Transformer):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def a(self, children):
                self.calls += 1
                return 'a_called'
            def b(self, children):
                self.calls += 1
                return 'b_called'

        t1 = CountingTransformer()
        lark1 = Lark(self.grammar, parser='lalr', transformer=t1)
        result1 = lark1.parse('a')
        calls_without = t1.calls

        t2 = CountingTransformer()
        lark2 = Lark(self.grammar, parser='lalr', transformer=t2)
        result2, _ = lark2.parse('a', coverage=True)
        calls_with = t2.calls

        self.assertEqual(calls_without, calls_with)
        self.assertEqual(result1, result2)

    def test_report_format(self):
        _, col = self.lark.parse('a', coverage=True)
        report = col.report()
        self.assertIsInstance(report, str)
        self.assertIn('Coverage Report', report)
        self.assertIn('Parse failed: False', report)
        self.assertIn('Total reductions:', report)
        self.assertIn('Unique rules reduced:', report)

    def test_failed_report_format(self):
        grammar = """
        start: "a" expr
        expr: "b" -> b | "c" -> c
        """
        lark = Lark(grammar, parser='lalr')
        try:
            lark.parse('abd', coverage=True)
        except UnexpectedInput as e:
            col = e.coverage
            report = col.report()
            self.assertIn('Parse failed: True', report)


class TestCoverageCollectorDirectly(unittest.TestCase):
    """Direct tests for CoverageCollector."""

    def test_initial_state(self):
        col = CoverageCollector()
        self.assertEqual(col.total_reductions, 0)
        self.assertEqual(col.rules, {})
        self.assertEqual(col.aliases, {})
        self.assertFalse(col.failed)

    def test_record_reduce(self):
        from lark.grammar import Rule, NonTerminal, Terminal
        col = CoverageCollector()
        rule = Rule(NonTerminal('start'), [Terminal('a')], alias='a')
        col.record_reduce(rule)
        self.assertEqual(col.total_reductions, 1)
        self.assertEqual(col.rules[rule], 1)
        self.assertEqual(col.aliases['a'], 1)

    def test_record_reduce_no_alias(self):
        from lark.grammar import Rule, NonTerminal, Terminal
        col = CoverageCollector()
        rule = Rule(NonTerminal('start'), [Terminal('a')])
        col.record_reduce(rule)
        self.assertEqual(col.total_reductions, 1)
        self.assertEqual(col.rules[rule], 1)
        self.assertEqual(col.aliases, {})

    def test_mark_failed(self):
        col = CoverageCollector()
        self.assertFalse(col.failed)
        col.mark_failed()
        self.assertTrue(col.failed)


if __name__ == '__main__':
    unittest.main()
