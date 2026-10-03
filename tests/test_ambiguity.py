"""Tests for lark.ambiguity (explicit expansion of ``_ambig`` forests)."""

from __future__ import absolute_import

import unittest
from unittest import TestCase

from lark import Lark, Tree, Token
from lark.ambiguity import AmbiguityCycleError, AmbiguityExpander, expand_ambiguities


def leaf(name):
    return Tree('leaf', [Token('WORD', name)])


def two_position_forest():
    """A forest with two ambiguous positions, two candidates each."""
    return Tree('root', [
        Tree('_ambig', [leaf('a1'), leaf('a2')]),
        Tree('_ambig', [leaf('b1'), leaf('b2')]),
    ])


def words(tree):
    """The token values of a tree, depth-first."""
    return [t.value for t in tree.scan_values(lambda v: isinstance(v, Token))]


class TestAmbiguityExpander(TestCase):
    def test_cartesian_product_fixed_order(self):
        # Two positions with two candidates each -> 4 trees, fixed order.
        expander = AmbiguityExpander(two_position_forest())
        self.assertEqual(expander.total, 4)
        self.assertEqual(len(expander), 4)
        self.assertFalse(expander.truncated)
        result = [words(t) for t in expander]
        self.assertEqual(result, [
            ['a1', 'b1'],
            ['a1', 'b2'],
            ['a2', 'b1'],
            ['a2', 'b2'],
        ])
        self.assertEqual(expander.count, 4)

    def test_max_expansions_truncates(self):
        # A cap of 2 yields only the first two trees, and reports truncation.
        expander = AmbiguityExpander(two_position_forest(), max_expansions=2)
        self.assertEqual(expander.total, 4)
        self.assertEqual(expander.limit, 2)
        self.assertTrue(expander.truncated)
        result = [words(t) for t in expander]
        self.assertEqual(result, [['a1', 'b1'], ['a1', 'b2']])

    def test_max_expansions_above_total_is_not_truncated(self):
        expander = AmbiguityExpander(two_position_forest(), max_expansions=10)
        self.assertFalse(expander.truncated)
        self.assertEqual(len(list(expander)), 4)

    def test_max_expansions_zero(self):
        expander = AmbiguityExpander(two_position_forest(), max_expansions=0)
        self.assertTrue(expander.truncated)
        self.assertEqual(list(expander), [])

    def test_max_expansions_must_not_be_negative(self):
        self.assertRaises(ValueError, AmbiguityExpander, two_position_forest(), max_expansions=-1)

    def test_results_are_independent(self):
        # Mutating an earlier result must not affect the later ones.
        trees = list(AmbiguityExpander(two_position_forest()))
        self.assertEqual(len({id(t.children) for t in trees}), 4)  # distinct child lists
        snapshot = [words(t) for t in trees]
        trees[0].children.append(Tree('extra', []))
        trees[0].children[0].children.append(Token('WORD', 'mutated'))
        trees[0].data = 'mutated'
        for t, expected in zip(trees[1:], snapshot[1:]):
            self.assertEqual(t.data, 'root')
            self.assertEqual(words(t), expected)

    def test_tree_and_token_metadata_preserved(self):
        tok = Token('WORD', 'x', line=3, column=7)
        inner = Tree('inner', [tok])
        inner.meta.line = 11
        inner.meta.empty = False
        forest = Tree('root', [Tree('_ambig', [inner, leaf('other')])])
        first, second = list(AmbiguityExpander(forest))
        # The chosen alternative keeps its meta, and the Token is kept as-is.
        self.assertEqual(first.children[0].meta.line, 11)
        self.assertIs(first.children[0].children[0], tok)
        self.assertEqual((first.children[0].children[0].line,
                          first.children[0].children[0].column), (3, 7))
        self.assertEqual(second.children[0].children[0].value, 'other')

    def test_shared_subtree_is_not_a_cycle(self):
        # The same subtree object referenced from two places (a DAG) must
        # expand normally; only genuine cycles are rejected.
        shared = Tree('_ambig', [leaf('x'), leaf('y')])
        forest = Tree('root', [shared, shared])
        expander = AmbiguityExpander(forest)
        self.assertEqual(expander.total, 4)
        result = [words(t) for t in expander]
        self.assertEqual(result, [['x', 'x'], ['x', 'y'], ['y', 'x'], ['y', 'y']])

    def test_shared_subtree_expands_to_independent_copies(self):
        shared = Tree('_ambig', [leaf('x'), leaf('y')])
        forest = Tree('root', [shared, shared])
        first = next(iter(AmbiguityExpander(forest)))
        self.assertIsNot(first.children[0], first.children[1])
        first.children[0].children.append(Token('WORD', 'mutated'))
        self.assertEqual(words(first.children[1]), ['x'])

    def test_genuine_cycle_is_rejected(self):
        loop = Tree('loop', [])
        loop.children.append(Tree('_ambig', [loop]))
        self.assertRaises(AmbiguityCycleError, AmbiguityExpander, loop)

    def test_direct_self_cycle_is_rejected(self):
        loop = Tree('_ambig', [])
        loop.children.append(loop)
        self.assertRaises(AmbiguityCycleError, AmbiguityExpander, loop)

    def test_early_close_stops_expansion(self):
        # A forest with a huge number of expansions: closing early must not
        # keep expanding, and iteration must stop immediately.
        forest = leaf('z')
        for _ in range(60):  # one more binary choice per level -> 2**60 expansions
            forest = Tree('wrap', [Tree('_ambig', [leaf('p'), leaf('q')]), forest])
        expander = AmbiguityExpander(forest)
        self.assertEqual(expander.total, 2 ** 60)
        next(iter(expander))
        self.assertEqual(expander.count, 1)
        expander.close()
        self.assertRaises(StopIteration, next, expander)
        self.assertRaises(StopIteration, next, expander)
        # Context-manager use closes the expander as well.
        with expand_ambiguities(forest) as exp:
            next(iter(exp))
        self.assertRaises(StopIteration, next, exp)

    def test_no_ambiguity_yields_single_copy(self):
        tree = Tree('root', [leaf('only')])
        expander = AmbiguityExpander(tree)
        self.assertEqual(expander.total, 1)
        self.assertFalse(expander.truncated)
        (result,) = expander
        self.assertEqual(result, tree)
        self.assertIsNot(result, tree)

    def test_empty_ambig_yields_nothing(self):
        expander = AmbiguityExpander(Tree('root', [Tree('_ambig', [])]))
        self.assertEqual(expander.total, 0)
        self.assertEqual(list(expander), [])

    def test_nested_ambig(self):
        forest = Tree('_ambig', [
            Tree('_ambig', [leaf('a'), leaf('b')]),
            leaf('c'),
        ])
        result = [words(t) for t in expand_ambiguities(forest)]
        self.assertEqual(result, [['a'], ['b'], ['c']])

    def test_earley_explicit_integration(self):
        grammar = r'''
            start: e
            e: e PLUS e
             | A
            PLUS: "+"
            A: "a"
        '''
        parser = Lark(grammar, parser='earley', ambiguity='explicit')
        forest = parser.parse('a+a+a')
        self.assertIn('_ambig', forest.pretty())
        expander = AmbiguityExpander(forest)
        self.assertEqual(expander.total, 2)
        trees = list(expander)
        self.assertEqual(len(trees), 2)
        for t in trees:
            self.assertNotIn('_ambig', t.pretty())
        self.assertNotEqual(trees[0], trees[1])
        # Both parses are distinct, fully expanded trees of the same input.
        for t in trees:
            self.assertEqual([tok.value for tok in t.scan_values(lambda v: isinstance(v, Token))],
                             ['a', '+', 'a', '+', 'a'])


if __name__ == '__main__':
    unittest.main()
