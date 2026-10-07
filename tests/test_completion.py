# -*- coding: utf-8 -*-
"""Tests for ``lark.completion`` - completion analysis for LALR parsers."""
from __future__ import absolute_import

import unittest
from unittest import TestCase

from lark import Lark
from lark.completion import CompletionAnalyzer, CompletionResult, CompletionCandidate, RulePosition
from lark.exceptions import ConfigurationError


QUERY_GRAMMAR = r"""
start: query
query: SELECT columns FROM table (WHERE condition)?
columns: STAR | column (COMMA column)*
column: NAME
table: NAME
condition: NAME EQ value
value: NUMBER | STRING

SELECT.2: "SELECT"
FROM.2: "FROM"
WHERE.2: "WHERE"
STAR: "*"
COMMA: ","
EQ: "="
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
%import common.NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
"""


class TestCompletion(TestCase):
    def setUp(self):
        self.parser = Lark(QUERY_GRAMMAR, parser='lalr')
        self.analyzer = CompletionAnalyzer(self.parser)

    def test_empty_input(self):
        result = self.analyzer.analyze("")
        self.assertIsInstance(result, CompletionResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.parsed_text, "")
        self.assertEqual(result.partial_token, "")
        self.assertIsNone(result.unexpected_token)
        self.assertFalse(result.accepts_eof)
        # The only way to start a query is the SELECT keyword
        self.assertEqual(result.token_types, {"SELECT"})
        self.assertEqual(result.literals, {"SELECT": "SELECT"})
        # The candidate points at the rule position that expects it
        candidate = result.get_candidate("SELECT")
        self.assertIsInstance(candidate, CompletionCandidate)
        self.assertTrue(candidate.positions)
        for pos in candidate.positions:
            self.assertIsInstance(pos, RulePosition)
            self.assertEqual(pos.expected_symbol, "SELECT")
        self.assertIn("query", {pos.rule_name for pos in candidate.positions})

    def test_after_select(self):
        result = self.analyzer.analyze("SELECT ")
        self.assertTrue(result.ok)
        self.assertEqual(result.parsed_text, "SELECT ")
        self.assertFalse(result.accepts_eof)
        # After SELECT we expect either '*' or a column name
        self.assertEqual(result.token_types, {"STAR", "NAME"})
        self.assertEqual(result.literals, {"STAR": "*"})
        # NAME is a regex terminal, so no literal can be determined
        self.assertIsNone(result.get_candidate("NAME").literal)
        # STAR is expected by the 'columns' rule
        star = result.get_candidate("STAR")
        self.assertIn("columns", {pos.rule_name for pos in star.positions})

    def test_complete_statement(self):
        for text in ("SELECT a FROM t", "SELECT a FROM t "):
            result = self.analyzer.analyze(text)
            self.assertTrue(result.ok)
            # The statement is complete: EOF is acceptable,
            # and an optional WHERE clause may follow
            self.assertTrue(result.accepts_eof)
            self.assertIn("$END", result.token_types)
            self.assertIn("WHERE", result.token_types)
            self.assertEqual(result.literals.get("WHERE"), "WHERE")
            # The candidates carry the rule positions that caused them
            self.assertTrue(result.get_candidate("$END").positions)
            # The returned state can resume and finish the parse normally
            tree = result.interactive_parser.resume_parse()
            self.assertEqual(tree, self.parser.parse("SELECT a FROM t"))

    def test_shared_token_positions(self):
        # Two different rules expect the same token at the same position
        grammar = r"""
            start: a_rule | b_rule
            a_rule: NAME "x"
            b_rule: NAME "y"
            NAME: /[a-z]+/
        """
        analyzer = CompletionAnalyzer(Lark(grammar, parser='lalr'))
        result = analyzer.analyze("")
        self.assertEqual(result.token_types, {"NAME"})
        candidate = result.get_candidate("NAME")
        rule_names = {pos.rule_name for pos in candidate.positions}
        self.assertEqual(rule_names, {"a_rule", "b_rule"})
        self.assertTrue(all(pos.index == 0 for pos in candidate.positions))

    def test_shared_token_in_query_grammar(self):
        # NAME is used by several rules; the reported positions reflect
        # every rule position that expects it at the current state
        result = self.analyzer.analyze("SELECT a FROM ")
        self.assertEqual(result.token_types, {"NAME"})
        candidate = result.get_candidate("NAME")
        self.assertIn("table", {pos.rule_name for pos in candidate.positions})

    def test_partial_keyword(self):
        # Half a keyword: the tail doesn't form a valid token here,
        # but the candidates still tell the caller what can be written
        result = self.analyzer.analyze("SEL")
        self.assertFalse(result.ok)
        self.assertEqual(result.token_types, {"SELECT"})
        self.assertEqual(result.literals, {"SELECT": "SELECT"})

        result = self.analyzer.analyze("SELECT a FR")
        self.assertFalse(result.ok)
        self.assertIn("FROM", result.token_types)
        self.assertEqual(result.literals.get("FROM"), "FROM")

    def test_partial_token_filters_candidates(self):
        # An unlexable tail narrows the candidates down by literal prefix
        grammar = r"""
            start: NAME ARROW NAME
            NAME: /[a-z]+/
            ARROW: "->"
            %import common.WS
            %ignore WS
        """
        analyzer = CompletionAnalyzer(Lark(grammar, parser='lalr'))
        result = analyzer.analyze("foo -")
        self.assertFalse(result.ok)
        self.assertEqual(result.parsed_text, "foo ")
        self.assertEqual(result.partial_token, "-")
        self.assertEqual(result.token_types, {"ARROW"})
        self.assertEqual(result.literals, {"ARROW": "->"})
        self.assertFalse(result.accepts_eof)

        # A tail that matches nothing leaves no literal candidates
        result = analyzer.analyze("foo %")
        self.assertEqual(result.partial_token, "%")
        self.assertNotIn("ARROW", result.token_types)

    def test_error_prefix(self):
        # A well-lexed prefix with a parse error: the candidates describe
        # what may appear instead of the offending token
        parser = Lark(QUERY_GRAMMAR, parser='lalr', lexer='basic')
        analyzer = CompletionAnalyzer(parser)
        result = analyzer.analyze("SELECT FROM t")
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.unexpected_token)
        self.assertEqual(result.unexpected_token.type, "FROM")
        self.assertEqual(result.parsed_text, "SELECT ")
        self.assertEqual(result.token_types, {"STAR", "NAME"})

    def test_consecutive_analysis_and_resume(self):
        # Two consecutive analyses of the same interactive parser
        ip = self.parser.parse_interactive("SELECT a FROM t")
        ip.exhaust_lexer()
        r1 = self.analyzer.analyze_interactive(ip)
        r2 = self.analyzer.analyze_interactive(ip)
        self.assertEqual(r1.token_types, r2.token_types)
        self.assertEqual(r1.token_types, {"$END", "WHERE"})
        self.assertTrue(r1.accepts_eof)
        # The original state is untouched and can keep parsing normally
        self.assertEqual(ip.accepts(), {"$END", "WHERE"})
        tree = ip.resume_parse()
        self.assertEqual(tree, self.parser.parse("SELECT a FROM t"))

        # Two consecutive text analyses give consistent results,
        # and the Lark instance is unaffected by the analysis
        r3 = self.analyzer.analyze("SELECT ")
        r4 = self.analyzer.analyze("SELECT ")
        self.assertEqual(r3.token_types, r4.token_types)
        self.assertEqual(r3.token_types, {"STAR", "NAME"})
        self.assertEqual(
            self.parser.parse("SELECT x FROM y WHERE z = 1").children[0].children[0],
            "SELECT",
        )

    def test_debug_parser(self):
        # Also works with a parser built in debug mode (non-integer states)
        parser = Lark(QUERY_GRAMMAR, parser='lalr', debug=True)
        analyzer = CompletionAnalyzer(parser)
        result = analyzer.analyze("SELECT ")
        self.assertEqual(result.token_types, {"STAR", "NAME"})

    def test_non_lalr_rejected(self):
        parser = Lark(QUERY_GRAMMAR, parser='earley')
        self.assertRaises(ConfigurationError, CompletionAnalyzer, parser)


if __name__ == '__main__':
    unittest.main()
