"""Tests for lark.error_recovery (InteractiveParser-based repair suggestions)."""

from copy import copy
from unittest import TestCase, main

from lark import Lark, Token, UnexpectedToken
from lark.visitors import Transformer
from lark.error_recovery import RecoveryReport, suggest_error_recovery


GRAMMAR = r"""
?start: expr
?expr: term (PLUS term)*
?term: factor (MUL factor)*
?factor: NUMBER | LPAR expr RPAR
NUMBER: /\d+/
PLUS: "+"
MUL: "*"
LPAR: "("
RPAR: ")"
%ignore " "
"""

CANDIDATES = [Token('RPAR', ')'), Token('PLUS', '+'), Token('MUL', '*'), Token('NUMBER', '0')]


def remaining_from(ip, first_token):
    """The failing token plus everything still left in the lexer (non-destructive)."""
    rest = list(copy(ip.lexer_thread).lex(ip.parser_state))
    return [first_token] + rest


class TestErrorRecovery(TestCase):
    def setUp(self):
        self.parser = Lark(GRAMMAR, parser='lalr')

    def fail(self, text):
        try:
            self.parser.parse(text)
        except UnexpectedToken as e:
            return e
        self.fail("Expected UnexpectedToken for %r" % text)

    def test_insert_missing_rparen(self):
        e = self.fail("1 + (2 * 3")
        self.assertEqual(e.token.type, '$END')
        ip = e.interactive_parser
        stack_before = list(ip.parser_state.state_stack)

        report = suggest_error_recovery(ip, [e.token], insert_candidates=CANDIDATES)

        self.assertEqual(report.status, RecoveryReport.COMPLETE)
        self.assertTrue(report.ok)
        best = report.best
        self.assertEqual(best.cost, 1.0)
        self.assertEqual(len(best.actions), 1)
        action, = best.actions
        self.assertEqual((action.kind, action.token_type, action.position), ('insert', 'RPAR', 0))

        # Replaying the suggestion succeeds and produces the expected tree
        tree = best.replay(ip)
        self.assertEqual(tree, self.parser.parse("1 + (2 * 3)"))

        # The search must not have mutated the original parser state
        self.assertEqual(stack_before, ip.parser_state.state_stack)

    def test_delete_extra_operator(self):
        e = self.fail("1 + + 2")
        self.assertEqual(e.token.type, 'PLUS')
        ip = e.interactive_parser

        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=CANDIDATES)

        self.assertEqual(report.status, RecoveryReport.COMPLETE)
        best = report.best
        self.assertEqual(best.cost, 1.0)
        action, = best.actions
        self.assertEqual((action.kind, action.token_type, action.position), ('delete', 'PLUS', 0))

        tree = best.replay(ip)
        self.assertEqual(tree, self.parser.parse("1 + 2"))

    def test_equal_cost_suggestions_are_stably_ordered(self):
        # "1 2" can be repaired at cost 1 in three ways: insert MUL, delete
        # the NUMBER, or insert PLUS. Equal-cost suggestions must be ordered
        # deterministically by token type and position.
        e = self.fail("1 2")
        ip = e.interactive_parser

        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=CANDIDATES,
                                        max_suggestions=3)

        self.assertEqual(report.status, RecoveryReport.COMPLETE)
        self.assertEqual(len(report.suggestions), 3)
        self.assertEqual([s.cost for s in report.suggestions], [1.0, 1.0, 1.0])
        self.assertEqual(
            [[(a.kind, a.token_type, a.position) for a in s.actions] for s in report.suggestions],
            [
                [('insert', 'MUL', 0)],
                [('delete', 'NUMBER', 0)],
                [('insert', 'PLUS', 0)],
            ])
        # Every reported suggestion actually replays successfully
        self.assertEqual(report.suggestions[0].replay(ip), self.parser.parse("1 * 2"))
        self.assertEqual(report.suggestions[1].replay(ip), self.parser.parse("1"))
        self.assertEqual(report.suggestions[2].replay(ip), self.parser.parse("1 + 2"))

    def test_error_at_eof(self):
        # The error token is $END and there is nothing left to delete;
        # only an insertion at end-of-input can repair the parse.
        e = self.fail("2 *")
        self.assertEqual(e.token.type, '$END')
        ip = e.interactive_parser

        report = suggest_error_recovery(ip, [e.token], insert_candidates=CANDIDATES)

        self.assertEqual(report.status, RecoveryReport.COMPLETE)
        action, = report.best.actions
        self.assertEqual((action.kind, action.token_type), ('insert', 'NUMBER'))
        self.assertEqual(report.best.replay(ip), self.parser.parse("2 * 0"))

    def test_empty_candidates_and_no_solution(self):
        # No insertion candidates and nothing deletable: genuinely unsolvable
        # within the cost bound, reported as complete with no suggestions.
        e = self.fail("1 + (2 * 3")
        report = suggest_error_recovery(e.interactive_parser, [e.token],
                                        insert_candidates=[])
        self.assertEqual(report.status, RecoveryReport.COMPLETE)
        self.assertFalse(report.ok)
        self.assertIsNone(report.best)
        self.assertEqual(report.suggestions, [])

    def test_deletion_only_when_no_candidates(self):
        # Deletions are always allowed, even with an empty insert whitelist.
        e = self.fail("1 + 2 )")
        self.assertEqual(e.token.type, 'RPAR')
        ip = e.interactive_parser

        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=[])
        self.assertTrue(report.ok)
        action, = report.best.actions
        self.assertEqual((action.kind, action.token_type), ('delete', 'RPAR'))
        self.assertEqual(report.best.replay(ip), self.parser.parse("1 + 2"))

    def test_configurable_costs(self):
        # If deletion is cheaper than insertion, "1 2" prefers deleting.
        e = self.fail("1 2")
        ip = e.interactive_parser
        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=CANDIDATES,
                                        insert_cost=2.0, delete_cost=1.0)
        action, = report.best.actions
        self.assertEqual(action.kind, 'delete')
        self.assertEqual(report.best.cost, 1.0)

        # If insertion is cheaper, inserting an operator wins.
        e = self.fail("1 2")
        ip = e.interactive_parser
        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=CANDIDATES,
                                        insert_cost=1.0, delete_cost=2.0)
        action, = report.best.actions
        self.assertEqual(action.kind, 'insert')
        self.assertEqual(report.best.cost, 1.0)

    def test_state_budget_reports_incomplete(self):
        e = self.fail("1 2")
        ip = e.interactive_parser
        report = suggest_error_recovery(ip, remaining_from(ip, e.token),
                                        insert_candidates=CANDIDATES,
                                        state_budget=0)
        # Budget exhausted: must not claim completeness or "no solution".
        self.assertEqual(report.status, RecoveryReport.INCOMPLETE)
        self.assertEqual(report.suggestions, [])

    def test_remaining_tokens_default(self):
        # remaining_tokens=None lexes the rest of the input non-destructively.
        e = self.fail("1 + (2 * 3")
        ip = e.interactive_parser
        report = suggest_error_recovery(ip, insert_candidates=CANDIDATES)
        self.assertTrue(report.ok)
        action, = report.best.actions
        self.assertEqual((action.kind, action.token_type), ('insert', 'RPAR'))

    def test_transformer_runs_only_on_final_replay(self):
        # User callbacks (an embedded, counting transformer) must not fire
        # during the search; only the final replay of the chosen suggestion
        # executes them, exactly once.
        class Calc(Transformer):
            def __init__(self):
                self.calls = 0

            def number(self, c):
                self.calls += 1
                return int(c[0])

            def add(self, c):
                self.calls += 1
                return c[0] + c[2]

            def mul(self, c):
                self.calls += 1
                return c[0] * c[2]

            def factor(self, c):
                self.calls += 1
                return c[1]

        grammar = r"""
        ?start: expr
        ?expr: expr PLUS term -> add
             | term
        ?term: term MUL factor -> mul
             | factor
        ?factor: NUMBER -> number
               | LPAR expr RPAR
        NUMBER: /\d+/
        PLUS: "+"
        MUL: "*"
        LPAR: "("
        RPAR: ")"
        %ignore " "
        """
        transformer = Calc()
        parser = Lark(grammar, parser='lalr', transformer=transformer)

        try:
            parser.parse("1 + (2 * 3")
        except UnexpectedToken as e:
            ip = e.interactive_parser
            error_token = e.token

        calls_after_error = transformer.calls

        report = suggest_error_recovery(ip, [error_token], insert_candidates=CANDIDATES,
                                        max_suggestions=3)
        self.assertTrue(report.ok)
        # Searching and verifying suggestions ran no user callbacks
        self.assertEqual(transformer.calls, calls_after_error)

        result = report.best.replay(ip)
        self.assertEqual(result, 7)
        # The replay ran the callbacks exactly as a clean parse would
        clean = Calc()
        Lark(grammar, parser='lalr', transformer=clean).parse("1 + (2 * 3)")
        self.assertEqual(transformer.calls, clean.calls)


if __name__ == '__main__':
    main()
