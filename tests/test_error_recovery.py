# -*- coding: utf-8 -*-

from unittest import TestCase, main

from lark import Lark, Token, Tree, UnexpectedToken
from lark.error_recovery import (
    RepairKind,
    RepairSuggester,
    SearchStatus,
)
from lark.visitors import Transformer


GRAMMAR = r'''
    ?start: expr
    ?expr: expr "+" term | term
    ?term: NUMBER | "(" expr ")"
    %import common.NUMBER
    %import common.WS
    %ignore WS
'''

INSERT_CANDIDATES = [
    Token('NUMBER', '0'),
    Token('LPAR', '('),
    Token('RPAR', ')'),
    Token('PLUS', '+'),
]


def parse_error(parser, text):
    try:
        parser.parse(text)
    except UnexpectedToken as e:
        return e
    raise AssertionError('Expected UnexpectedToken for %r' % text)


class TestErrorRecovery(TestCase):
    def setUp(self):
        self.parser = Lark(GRAMMAR, parser='lalr')

    def test_missing_right_paren_insert(self):
        # '1+(2' fails at end-of-input; the cheapest repair is inserting ')'
        e = parse_error(self.parser, '1+(2')
        self.assertEqual(e.token.type, '$END')
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, INSERT_CANDIDATES)
        self.assertIs(result.status, SearchStatus.COMPLETE)
        self.assertTrue(result.complete)
        self.assertTrue(result.repairs)
        repair = result.repairs[0]
        self.assertEqual(repair.cost, 1.0)
        self.assertEqual(len(repair.actions), 1)
        action = repair.actions[0]
        self.assertIs(action.kind, RepairKind.INSERT)
        self.assertEqual(action.token.type, 'RPAR')
        self.assertTrue(repair.verified)
        # Replaying the repair produces the expected tree
        tree = repair.apply(e.interactive_parser)
        self.assertEqual(tree, self.parser.parse('1+(2)'))

    def test_extra_operator_delete(self):
        # '1 + + 2' fails on the second '+'; deleting it repairs the parse
        e = parse_error(self.parser, '1 + + 2')
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, [Token('NUMBER', '0')])
        self.assertIs(result.status, SearchStatus.COMPLETE)
        deletes = [r for r in result.repairs
                   if all(a.kind is RepairKind.DELETE for a in r.actions)]
        self.assertTrue(deletes)
        repair = deletes[0]
        self.assertEqual(repair.cost, 1.0)
        self.assertEqual(repair.actions[0].token.type, 'PLUS')
        tree = repair.apply(e.interactive_parser)
        self.assertEqual(tree, self.parser.parse('1 + 2'))

    def test_same_cost_multiple_solutions_stable_order(self):
        # Both 'insert NUMBER' and 'delete PLUS' fix '1 + + 2' at cost 1.
        e = parse_error(self.parser, '1 + + 2')
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, [Token('NUMBER', '0')])
        self.assertEqual(len(result.repairs), 2)
        costs = [r.cost for r in result.repairs]
        self.assertEqual(costs, [1.0, 1.0])
        # Stable tie-break: sorted by token type, then position
        first, second = result.repairs
        self.assertIs(first.actions[0].kind, RepairKind.INSERT)
        self.assertEqual(first.actions[0].token.type, 'NUMBER')
        self.assertIs(second.actions[0].kind, RepairKind.DELETE)
        self.assertEqual(second.actions[0].token.type, 'PLUS')
        # Every suggested repair replays successfully
        for repair in result.repairs:
            self.assertTrue(repair.verified)
            repair.apply(e.interactive_parser)

    def test_eof_failure_empty_remaining(self):
        # Failure at EOF: the remaining token stream is empty
        e = parse_error(self.parser, '(1+2')
        self.assertEqual(e.token.type, '$END')
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, INSERT_CANDIDATES)
        self.assertIs(result.status, SearchStatus.COMPLETE)
        self.assertTrue(result.repairs)
        repair = result.repairs[0]
        self.assertEqual([a.kind for a in repair.actions], [RepairKind.INSERT])
        self.assertEqual(repair.actions[0].token.type, 'RPAR')
        self.assertEqual(repair.apply(e.interactive_parser),
                         self.parser.parse('(1+2)'))

    def test_empty_candidates(self):
        # With an empty insert whitelist, only deletions are possible.
        # '(1+2' cannot be repaired by deletion within max_cost: the result
        # is COMPLETE (proven unsolvable within the cost bound), not INCOMPLETE.
        e = parse_error(self.parser, '(1+2')
        suggester = RepairSuggester(max_cost=2)
        result = suggester.suggest_from_error(e, [])
        self.assertIs(result.status, SearchStatus.COMPLETE)
        self.assertEqual(result.repairs, [])
        # ...while '1 + + 2' is repaired by deletion alone
        e = parse_error(self.parser, '1 + + 2')
        result = suggester.suggest_from_error(e, [])
        self.assertIs(result.status, SearchStatus.COMPLETE)
        self.assertEqual(len(result.repairs), 1)
        self.assertIs(result.repairs[0].actions[0].kind, RepairKind.DELETE)

    def test_incomplete_status_on_state_budget(self):
        # A tiny state budget must yield INCOMPLETE, never a false 'no solution'
        e = parse_error(self.parser, '1+(2')
        suggester = RepairSuggester(max_states=1)
        result = suggester.suggest_from_error(e, INSERT_CANDIDATES)
        self.assertIs(result.status, SearchStatus.INCOMPLETE)
        self.assertFalse(result.complete)

    def test_configurable_costs(self):
        # With deletion cheaper than insertion, the delete repair wins
        e = parse_error(self.parser, '1 + + 2')
        suggester = RepairSuggester(insert_cost=5.0, delete_cost=1.0)
        result = suggester.suggest_from_error(e, [Token('NUMBER', '0')])
        self.assertEqual(len(result.repairs), 1)
        self.assertIs(result.repairs[0].actions[0].kind, RepairKind.DELETE)
        # With insertion cheaper, the insert repair wins
        suggester = RepairSuggester(insert_cost=1.0, delete_cost=5.0)
        result = suggester.suggest_from_error(e, [Token('NUMBER', '0')])
        self.assertEqual(len(result.repairs), 1)
        self.assertIs(result.repairs[0].actions[0].kind, RepairKind.INSERT)

    def test_string_candidates(self):
        # Token-type names are accepted as insert candidates
        e = parse_error(self.parser, '1+(2')
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, ['RPAR'])
        self.assertTrue(result.repairs)
        self.assertEqual(result.repairs[0].actions[0].token.type, 'RPAR')

    def test_search_has_no_side_effects(self):
        # Searching must not mutate the failed parser state, nor run user
        # transformer callbacks.  Callbacks run only when the chosen repair
        # is applied, and only for that single repair path.
        counter = []

        class CountingTransformer(Transformer):
            def expr(self, children):
                counter.append('expr')
                return Tree('expr', children)

            def term(self, children):
                counter.append('term')
                return Tree('term', children)

        parser = Lark(GRAMMAR, parser='lalr', transformer=CountingTransformer())
        e = parse_error(parser, '1+(2')
        stack_before = list(e.interactive_parser.parser_state.state_stack)
        suggester = RepairSuggester()
        result = suggester.suggest_from_error(e, INSERT_CANDIDATES)
        self.assertTrue(result.repairs)
        # No user callbacks fired during the search
        self.assertEqual(counter, [])
        # The failed parser state is untouched
        self.assertEqual(e.interactive_parser.parser_state.state_stack,
                         stack_before)
        # Applying the chosen repair runs the transformer exactly once
        # (for that repair path only)
        tree = result.repairs[0].apply(e.interactive_parser)
        expected = parser.parse('1+(2)')
        self.assertEqual(tree, expected)
        # ('term' is inlined via '?term', so only 'expr' callbacks fire)
        self.assertEqual(counter, ['expr', 'expr'])

    def test_trailing_end_token_ignored(self):
        # A trailing $END in the remaining stream is normalized away
        e = parse_error(self.parser, '1 + + 2')
        ip = e.interactive_parser
        remaining = [e.token, Token('NUMBER', '2'), Token('$END', '')]
        suggester = RepairSuggester()
        result = suggester.suggest(ip, remaining, [])
        self.assertTrue(result.repairs)
        self.assertIs(result.repairs[0].actions[0].kind, RepairKind.DELETE)


if __name__ == '__main__':
    main()
