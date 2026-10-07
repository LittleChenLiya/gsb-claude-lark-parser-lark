# -*- coding: utf-8 -*-
"""Tests for lark.completion (completion analysis on top of the LALR InteractiveParser)."""
from __future__ import absolute_import

import unittest

from lark import Lark, Token
from lark.completion import CompletionAnalyzer, CompletionCandidate, CompletionResult, RulePosition
from lark.exceptions import ConfigurationError, UnexpectedInput


QUERY_GRAMMAR = r"""
?start: select_stmt
select_stmt: SELECT select_list FROM table_name where_clause?
select_list: STAR | column_name (COMMA column_name)*
where_clause: WHERE condition
condition: column_name EQ value
column_name: NAME
table_name: NAME
value: NAME | NUMBER | func_call
func_call: NAME LPAREN NAME RPAREN

SELECT: "select"i
FROM: "from"i
WHERE: "where"i
STAR: "*"
COMMA: ","
EQ: "="
LPAREN: "("
RPAREN: ")"
%import common.CNAME -> NAME
%import common.NUMBER
%import common.WS
%ignore WS
"""


class TestCompletionAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = Lark(QUERY_GRAMMAR, parser='lalr')
        cls.analyzer = CompletionAnalyzer(cls.parser)

    def test_empty_input(self):
        # Empty prefix: the only possible next step is the SELECT keyword
        res = self.analyzer.analyze('')
        self.assertIsNone(res.error)
        self.assertTrue(res.input_exhausted)
        self.assertFalse(res.accepts_eof)
        self.assertEqual(res.tokens_consumed, ())
        self.assertEqual(res.token_types, ['SELECT'])

        select = res.candidate('SELECT')
        self.assertIsNotNone(select)
        # "SELECT" is a fixed-string terminal, so the literal is known
        self.assertEqual(select.literal, 'select')
        # The candidate is caused by the select_stmt rule(s), at position 0
        self.assertTrue(select.positions)
        for pos in select.positions:
            self.assertIsInstance(pos, RulePosition)
            self.assertEqual(pos.origin, 'select_stmt')
            self.assertEqual(pos.index, 0)
            self.assertEqual(pos.expected_symbol, 'SELECT')

    def test_after_select(self):
        # After "SELECT", the select list must follow: a star or a column name
        res = self.analyzer.analyze('SELECT ')
        self.assertIsNone(res.error)
        self.assertFalse(res.accepts_eof)
        self.assertEqual([t.type for t in res.tokens_consumed], ['SELECT'])
        self.assertEqual(set(res.token_types), {'STAR', 'NAME'})

        star = res.candidate('STAR')
        self.assertEqual(star.literal, '*')
        self.assertEqual([p.origin for p in star.positions], ['select_list'])

        name = res.candidate('NAME')
        # NAME is a regexp terminal: no directly determinable literal
        self.assertIsNone(name.literal)
        self.assertEqual([p.origin for p in name.positions], ['column_name'])

    def test_complete_statement(self):
        # A complete statement: input may end here, or continue with WHERE
        res = self.analyzer.analyze('SELECT a FROM t')
        self.assertIsNone(res.error)
        self.assertTrue(res.input_exhausted)
        self.assertTrue(res.accepts_eof)
        self.assertEqual(set(res.token_types), {'WHERE', '$END'})

        where = res.candidate('WHERE')
        self.assertEqual(where.literal, 'where')
        self.assertEqual([p.origin for p in where.positions], ['where_clause'])

        eof = res.candidate('$END')
        self.assertIsNotNone(eof)
        self.assertIsNone(eof.literal)
        # EOF is acceptable because select_stmt is complete here
        self.assertTrue(eof.positions)
        self.assertIn('select_stmt', {p.origin for p in eof.positions})

        # A fully exhausted statement only accepts EOF
        res2 = self.analyzer.analyze('SELECT a FROM t WHERE x = 1')
        self.assertIsNone(res2.error)
        self.assertTrue(res2.accepts_eof)
        self.assertEqual(res2.token_types, ['$END'])

    def test_token_shared_by_multiple_rules(self):
        # After "=", a NAME may start either a plain value or a function call:
        # a single token type is expected by several rule positions
        res = self.analyzer.analyze('SELECT a FROM t WHERE x = ')
        self.assertIsNone(res.error)
        self.assertEqual(set(res.token_types), {'NAME', 'NUMBER'})

        name = res.candidate('NAME')
        origins = {p.origin for p in name.positions}
        self.assertEqual(origins, {'value', 'func_call'})

        number = res.candidate('NUMBER')
        self.assertEqual([p.origin for p in number.positions], ['value'])

    def test_half_keyword(self):
        # "SEL" is half of the SELECT keyword: the prefix cannot be consumed,
        # but the candidates still tell the caller what may be written next
        for prefix in ('SEL', 'SELEC'):
            res = self.analyzer.analyze(prefix)
            self.assertIsNotNone(res.error)
            self.assertIsInstance(res.error, UnexpectedInput)
            self.assertFalse(res.input_exhausted)
            self.assertEqual(res.tokens_consumed, ())
            self.assertEqual(res.token_types, ['SELECT'])
            self.assertEqual(res.candidate('SELECT').literal, 'select')

    def test_error_prefix(self):
        # "SELECT * WHERE" is invalid: WHERE cannot follow the select list.
        # The analyzer consumes the valid part and reports candidates at the
        # point of failure.
        res = self.analyzer.analyze('SELECT * WHERE')
        self.assertIsNotNone(res.error)
        self.assertIsInstance(res.error, UnexpectedInput)
        self.assertFalse(res.input_exhausted)
        self.assertEqual([t.type for t in res.tokens_consumed], ['SELECT', 'STAR'])
        self.assertEqual(res.token_types, ['FROM'])
        self.assertEqual(res.candidate('FROM').literal, 'from')

    def test_consecutive_analyses(self):
        # Two consecutive analyses on the same analyzer are independent
        res1 = self.analyzer.analyze('')
        res2 = self.analyzer.analyze('SELECT ')
        self.assertEqual(res1.token_types, ['SELECT'])
        self.assertEqual(set(res2.token_types), {'STAR', 'NAME'})
        # Repeating the first analysis gives the same result again
        res3 = self.analyzer.analyze('')
        self.assertEqual(res3, res1)

    def test_parser_state_not_affected(self):
        # An interactive parser created before the analysis keeps working
        ip = self.parser.parse_interactive('SELECT a FROM t')
        self.analyzer.analyze('')
        self.analyzer.analyze('SELECT a FROM t WHERE x = ')
        tokens = ip.exhaust_lexer()
        self.assertEqual([t.type for t in tokens], ['SELECT', 'NAME', 'FROM', 'NAME'])
        tree = ip.feed_eof(tokens[-1])
        self.assertEqual(tree.data, 'select_stmt')

        # The Lark instance itself still parses normally
        tree = self.parser.parse('SELECT a, b FROM t WHERE x = f(y)')
        self.assertEqual(tree.data, 'select_stmt')

    def test_result_structure(self):
        res = self.analyzer.analyze('SELECT ')
        self.assertIsInstance(res, CompletionResult)
        for candidate in res.candidates:
            self.assertIsInstance(candidate, CompletionCandidate)
            self.assertIsInstance(candidate.token_type, str)
            for pos in candidate.positions:
                self.assertIsInstance(pos, RulePosition)
        # candidate() returns None for unknown token types
        self.assertIsNone(res.candidate('FROM'))

    def test_configuration_errors(self):
        # Only LALR parsers are supported
        earley = Lark(QUERY_GRAMMAR, parser='earley')
        self.assertRaises(ConfigurationError, CompletionAnalyzer, earley)
        # Unknown start rule
        self.assertRaises(ConfigurationError, CompletionAnalyzer, self.parser, start='nosuchrule')

    def test_eof_acceptable_on_empty_input(self):
        # A grammar that accepts the empty input: EOF is a candidate from the start
        parser = Lark("start: item*\nitem: NAME\n%import common.CNAME -> NAME\n%import common.WS\n%ignore WS\n",
                      parser='lalr')
        analyzer = CompletionAnalyzer(parser)
        res = analyzer.analyze('')
        self.assertIsNone(res.error)
        self.assertTrue(res.accepts_eof)
        self.assertEqual(set(res.token_types), {'NAME', '$END'})
        res = analyzer.analyze('foo bar')
        self.assertTrue(res.accepts_eof)
        self.assertEqual(set(res.token_types), {'NAME', '$END'})


if __name__ == '__main__':
    unittest.main()
