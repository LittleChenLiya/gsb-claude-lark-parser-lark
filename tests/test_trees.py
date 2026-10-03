from __future__ import absolute_import

import unittest
from functools import partial, reduce, partialmethod
from operator import add, mul
from unittest import TestCase
import copy
import pickle
import functools

from lark.tree import Tree, Meta
from lark.lexer import Token
from lark.visitors import Visitor, Visitor_Recursive, Transformer, Interpreter, visit_children_decor, v_args, Discard, Transformer_InPlace, \
    Transformer_InPlaceRecursive, Transformer_NonRecursive, merge_transformers, ExpandAmbiguities


class TestTrees(TestCase):
    def setUp(self):
        self.tree1 = Tree('a', [Tree(x, y) for x, y in zip('bcd', 'xyz')])
        self.tree2 = Tree('a', [
            Tree('b', [Token('T', 'x')]),
            Tree('c', [Token('T', 'y')]),
            Tree('d', [Tree('z', [Token('T', 'zz'), Tree('zzz', 'zzz')])]),
        ])

    def test_eq(self):
        assert self.tree1 == self.tree1
        assert self.tree1 != 0

    def test_copy(self):
        assert self.tree1 == copy.copy(self.tree1)

    def test_deepcopy(self):
        assert self.tree1 == copy.deepcopy(self.tree1)

    def test_pickle(self):
        s = copy.deepcopy(self.tree1)
        data = pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL)
        assert pickle.loads(data) == s

    def test_token_deepcopy_and_pickle_preserve_positions(self):
        # Token.__eq__ ignores positions, so a copy that drops end_line/end_column/end_pos
        # would still compare equal. Assert the positions survive deepcopy and pickling.
        t = Token('A', 'abc', start_pos=10, line=1, column=3, end_line=2, end_column=6, end_pos=13)
        copies = [copy.deepcopy(t), pickle.loads(pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL))]
        for t2 in copies:
            self.assertEqual((t2.start_pos, t2.line, t2.column), (10, 1, 3))
            self.assertEqual((t2.end_line, t2.end_column, t2.end_pos), (2, 6, 13))

    def test_repr_runnable(self):
        assert self.tree1 == eval(repr(self.tree1))

    def test_iter_subtrees(self):
        expected = [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z'),
                    Tree('a', [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')])]
        nodes = list(self.tree1.iter_subtrees())
        self.assertEqual(nodes, expected)

    def test_iter_subtrees_topdown(self):
        expected = [Tree('a', [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')]),
                    Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')]
        nodes = list(self.tree1.iter_subtrees_topdown())
        self.assertEqual(nodes, expected)

    def test_find_token(self):
        expected = [Token('T', 'x'), Token('T', 'y'), Token('T', 'zz')]
        tokens = list(self.tree2.find_token('T'))
        self.assertEqual(tokens, expected)

    def test_visitor(self):
        class Visitor1(Visitor):
            def __init__(self):
                self.nodes=[]

            def __default__(self,tree):
                self.nodes.append(tree)
        class Visitor1_Recursive(Visitor_Recursive):
            def __init__(self):
                self.nodes=[]

            def __default__(self,tree):
                self.nodes.append(tree)

        visitor1=Visitor1()
        visitor1_recursive=Visitor1_Recursive()

        expected_top_down = [Tree('a', [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')]),
                    Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')]
        expected_botton_up= [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z'),
                    Tree('a', [Tree('b', 'x'), Tree('c', 'y'), Tree('d', 'z')])]

        visitor1.visit(self.tree1)
        self.assertEqual(visitor1.nodes,expected_botton_up)

        visitor1_recursive.visit(self.tree1)
        self.assertEqual(visitor1_recursive.nodes,expected_botton_up)

        visitor1.nodes=[]
        visitor1_recursive.nodes=[]

        visitor1.visit_topdown(self.tree1)
        self.assertEqual(visitor1.nodes,expected_top_down)

        visitor1_recursive.visit_topdown(self.tree1)
        self.assertEqual(visitor1_recursive.nodes,expected_top_down)

    def test_interp(self):
        t = Tree('a', [Tree('b', []), Tree('c', []), 'd'])

        class Interp1(Interpreter):
            def a(self, tree):
                return self.visit_children(tree) + ['e']

            def b(self, tree):
                return 'B'

            def c(self, tree):
                return 'C'

        self.assertEqual(Interp1().visit(t), list('BCde'))

        class Interp2(Interpreter):
            @visit_children_decor
            def a(self, values):
                return values + ['e']

            def b(self, tree):
                return 'B'

            def c(self, tree):
                return 'C'

        self.assertEqual(Interp2().visit(t), list('BCde'))

        class Interp3(Interpreter):
            def b(self, tree):
                return 'B'

            def c(self, tree):
                return 'C'

        self.assertEqual(Interp3().visit(t), list('BCd'))

    def test_transformer(self):
        t = Tree('add', [Tree('sub', [Tree('i', ['3']), Tree('f', ['1.1'])]), Tree('i', ['1'])])

        class T(Transformer):
            i = v_args(inline=True)(int)
            f = v_args(inline=True)(float)

            sub = lambda self, values: values[0] - values[1]

            def add(self, values):
                return sum(values)

        res = T().transform(t)
        self.assertEqual(res, 2.9)

        @v_args(inline=True)
        class T(Transformer):
            i = int
            f = float
            sub = lambda self, a, b: a-b

            def add(self, a, b):
                return a + b


        res = T().transform(t)
        self.assertEqual(res, 2.9)


        @v_args(inline=True)
        class T(Transformer):
            i = int
            f = float
            from operator import sub, add

        res = T().transform(t)
        self.assertEqual(res, 2.9)

    def test_vargs(self):
        @v_args()
        class MyTransformer(Transformer):
            @staticmethod
            def integer(args):
                return 1 # some code here

            @classmethod
            def integer2(cls, args):
                return 2 # some code here

            hello = staticmethod(lambda args: 'hello')

        x = MyTransformer().transform( Tree('integer', [2]))
        self.assertEqual(x, 1)
        x = MyTransformer().transform( Tree('integer2', [2]))
        self.assertEqual(x, 2)
        x = MyTransformer().transform( Tree('hello', [2]))
        self.assertEqual(x, 'hello')

    def test_smart_decorator(self):
        class OtherClass:
            @staticmethod
            def ab_staticmethod(a, b):
                return (a, b)

            @classmethod
            def ab_classmethod(cls, a, b):
                assert cls is OtherClass, cls
                return (a, b)

            def ab_method(self, a, b):
                assert isinstance(self, OtherClass), self
                return (a, b)

        @v_args(meta=True)
        class OtherTransformer(Transformer):
            @staticmethod
            def ab_staticmethod(meta, children):
                return tuple(children)

            @classmethod
            def ab_classmethod(cls, meta, children):
                assert cls is OtherTransformer, cls
                return tuple(children)

            def ab_method(self, meta, children):
                assert isinstance(self, OtherTransformer), self
                return tuple(children)

        class CustomCallable:
            def __call__(self, *args, **kwargs):
                assert isinstance(self, CustomCallable)
                return args

        oc_instance = OtherClass()
        ot_instance = OtherTransformer()

        def ab_for_partialmethod(self, a, b):
            assert isinstance(self, TestCls)
            return a, b

        @v_args(inline=True)
        class TestCls(Transformer):
            @staticmethod
            def ab_staticmethod(a, b):
                return (a, b)

            @classmethod
            def ab_classmethod(cls, a, b):
                assert cls is TestCls
                return (a, b)

            def ab_method(self, a, b):
                assert isinstance(self, TestCls)
                return (a, b)

            oc_class_ab_staticmethod = oc_instance.ab_staticmethod
            oc_class_ab_classmethod = oc_instance.ab_classmethod

            oc_ab_staticmethod = oc_instance.ab_staticmethod
            oc_ab_classmethod = oc_instance.ab_classmethod
            oc_ab_method = oc_instance.ab_method

            ot_class_ab_staticmethod = ot_instance.ab_staticmethod
            ot_class_ab_classmethod = ot_instance.ab_classmethod

            ot_ab_staticmethod = ot_instance.ab_staticmethod
            ot_ab_classmethod = ot_instance.ab_classmethod
            ot_ab_method = ot_instance.ab_method

            ab_partialmethod = partialmethod(ab_for_partialmethod, 1)
            set_union = set(["a"]).union
            static_add = staticmethod(add)
            partial_reduce_mul = staticmethod(partial(reduce, mul))

            custom_callable = CustomCallable()

        test_instance = TestCls()
        expected = {
            "ab_classmethod": ([1, 2], (1, 2)),
            "ab_staticmethod": ([1, 2], (1, 2)),
            "ab_method": ([1, 2], (1, 2)),
            "oc_ab_classmethod": ([1, 2], (1, 2)),
            "oc_class_ab_classmethod": ([1, 2], (1, 2)),

            # AFAIK, these two cases are impossible to deal with. `oc_instance.ab_staticmethod` returns an actual
            # function object that is impossible to distinguish from a normally defined method.
            # (i.e. `staticmethod(f).__get__(?, ?) is f` is True)
            # "oc_ab_staticmethod": ([1, 2], (1, 2)),
            # "oc_class_ab_staticmethod": ([1, 2], (1, 2)),

            "oc_ab_method": ([1, 2], (1, 2)),
            "ot_ab_classmethod": ([1, 2], (1, 2)),
            "ot_class_ab_classmethod": ([1, 2], (1, 2)),

            # Same as above
            # "ot_ab_staticmethod": ([1, 2], (1, 2)),
            # "ot_class_ab_staticmethod": ([1, 2], (1, 2)),

            "ot_ab_method": ([1, 2], (1, 2)),
            "ab_partialmethod": ([2], (1, 2)),
            "custom_callable": ([1, 2], (1, 2)),
            "set_union": ([["b"], ["c"]], {"a", "b", "c"}),
            "static_add": ([1, 2], 3),
            "partial_reduce_mul": ([[1, 2]], 2),
        }
        non_static = {"ab_method", "ab_partialmethod"}
        for method_name, (children, expected_result) in expected.items():
            not_inline = "ot" in method_name
            result = test_instance.transform(Tree(method_name, children))
            self.assertEqual(result, expected_result)

            if not_inline:
                result = getattr(test_instance, method_name)(None, children)
            else:
                result = getattr(test_instance, method_name)(*children)
            self.assertEqual(result, expected_result)

            if method_name not in non_static:
                if not_inline:
                    result = getattr(TestCls, method_name)(None, children)
                else:
                    result = getattr(TestCls, method_name)(*children)
                self.assertEqual(result, expected_result)

    def test_vargs_set_name(self):
        # Test with cached_property if available. That actually uses __set_name__
        prop = getattr(functools, "cached_property", property)

        class T(Transformer):
            @v_args(inline=True)
            @prop  # Not sure why you would ever want to use a property here, but we support it
            def test(self):
                return lambda a, b: (self, a, b)

        t = T()
        self.assertEqual(t.transform(Tree("test", [1, 2])), (t, 1, 2))

    def test_inline_static(self):
        @v_args(inline=True)
        class T(Transformer):
            @staticmethod
            def test(a, b):
                return a + b
        x = T().transform(Tree('test', ['a', 'b']))
        self.assertEqual(x, 'ab')

    def test_vargs_override(self):
        t = Tree('add', [Tree('sub', [Tree('i', ['3']), Tree('f', ['1.1'])]), Tree('i', ['1'])])

        @v_args(inline=True)
        class T(Transformer):
            i = int
            f = float
            sub = lambda self, a, b: a-b

            not_a_method = {'other': 'stuff'}

            @v_args(inline=False)
            def add(self, values):
                return sum(values)

        res = T().transform(t)
        self.assertEqual(res, 2.9)

    def test_partial(self):

        tree = Tree("start", [Tree("a", ["test1"]), Tree("b", ["test2"])])

        def test(prefix, s, postfix):
            return prefix + s.upper() + postfix

        @v_args(inline=True)
        class T(Transformer):
            a = staticmethod(functools.partial(test, "@", postfix="!"))
            b = staticmethod(functools.partial(lambda s: s + "!"))

        res = T().transform(tree)
        assert res.children == ["@TEST1!", "test2!"]

    def test_discard(self):
        class MyTransformer(Transformer):
            def a(self, args):
                return 1 # some code here

            def b(cls, args):
                return Discard

        t = Tree('root', [
            Tree('b', []),
            Tree('a', []),
            Tree('b', []),
            Tree('c', []),
            Tree('b', []),
        ])
        t2 = Tree('root', [1, Tree('c', [])])

        x = MyTransformer().transform( t )
        self.assertEqual(x, t2)

    def test_transformer_variants(self):
        tree = Tree('start', [
            Tree('add', [Token('N', '1'), Token('N', '2'), Token('IGNORE_TOKEN', '4')]),
            Tree('add', [Token('N', '3'), Token('N', '4')]),
            Tree('ignore_tree', [Token('DO', 'NOT PANIC')]),
            ])
        for base in (Transformer, Transformer_InPlace, Transformer_NonRecursive, Transformer_InPlaceRecursive):
            class T(base):
                def add(self, children):
                    return sum(children)

                def N(self, token):
                    return int(token)

                def ignore_tree(self, children):
                    return Discard

                def IGNORE_TOKEN(self, token):
                    return Discard

            copied = copy.deepcopy(tree)
            result = T().transform(copied)
            self.assertEqual(result, Tree('start', [3, 7]))

    def test_transformer_discard_root(self):
        """Transformer_NonRecursive should return None (not crash) when the root node is discarded."""
        for base in (Transformer, Transformer_NonRecursive):
            class T(base):
                def start(self, children):
                    return Discard

            tree = Tree('start', [Token('A', 'x')])
            result = T().transform(tree)
            self.assertIsNone(result)

    def test_transformer_variants_discard_with_preceding_sibling(self):
        # A discarded node whose parent is preceded by another subtree must not
        # corrupt the tree. Transformer_NonRecursive used to mis-slice its stack
        # here and steal `keep` into `wrap` (dropping it from `start`).
        tree = Tree('start', [
            Tree('keep', [Token('T', 'x')]),
            Tree('wrap', [Tree('drop', [])]),
            ])
        expected = Tree('start', [Tree('keep', [Token('T', 'x')]), Tree('wrap', [])])
        for base in (Transformer, Transformer_InPlace, Transformer_NonRecursive, Transformer_InPlaceRecursive):
            class T(base):
                def drop(self, children):
                    return Discard

            result = T().transform(copy.deepcopy(tree))
            self.assertEqual(result, expected)

    def test_merge_transformers(self):
        tree = Tree('start', [
            Tree('main', [
                Token("A", '1'), Token("B", '2')
            ]),
            Tree("module__main", [
                Token("A", "2"), Token("B", "3")
            ])
        ])

        class T1(Transformer):
            A = int
            B = int
            main = sum
            start = list
            def module__main(self, children):
                return sum(children)

        class T2(Transformer):
            A = int
            B = int
            main = sum
            start = list

        class T3(Transformer):
            def main(self, children):
                return sum(children)

        class T4(Transformer):
            main = sum


        t1_res = T1().transform(tree)
        composed_res = merge_transformers(T2(), module=T3()).transform(tree)
        self.assertEqual(t1_res, composed_res)

        composed_res2 = merge_transformers(T2(), module=T4()).transform(tree)
        self.assertEqual(t1_res, composed_res2)

        with self.assertRaises(AttributeError):
            merge_transformers(T1(), module=T3())

    def test_transform_token(self):
        class MyTransformer(Transformer):
            def INT(self, value):
                return int(value)

        t = Token('INT', '123')
        assert MyTransformer().transform(t) == 123

        class MyTransformer(Transformer):
            def INT(self, value):
                return Discard

        assert MyTransformer().transform(t) is None

    def test_incorrect_use_of_decorators(self):
        class TestVisitor1(Visitor):
            @visit_children_decor
            def a(self, tree):
                pass

        for visit_method in [
            TestVisitor1().visit,
            TestVisitor1().visit_topdown,
        ]:
            with self.assertRaises(TypeError) as e:
                visit_method(self.tree1)
            self.assertIn("visit_children_decor", str(e.exception))
            self.assertIn("Interpreter", str(e.exception))


def _two_position_fixture():
    """Two ``_ambig`` positions, each with 2 candidates (4 combinations)."""
    return Tree('start', [
        Tree('_ambig', [Tree('a', ['1']), Tree('a', ['2'])]),
        Tree('_ambig', [Tree('b', ['3']), Tree('b', ['4'])]),
    ])


def _four_expected():
    return [
        Tree('start', [Tree('a', ['1']), Tree('b', ['3'])]),
        Tree('start', [Tree('a', ['1']), Tree('b', ['4'])]),
        Tree('start', [Tree('a', ['2']), Tree('b', ['3'])]),
        Tree('start', [Tree('a', ['2']), Tree('b', ['4'])]),
    ]


class TestExpandAmbiguities(TestCase):
    def test_two_positions_yield_four_trees_in_fixed_order(self):
        expander = ExpandAmbiguities(_two_position_fixture())
        trees = list(expander)
        self.assertEqual(expander.count, 4)
        self.assertFalse(expander.truncated)
        self.assertEqual(trees, _four_expected())

    def test_max_trees_caps_and_marks_truncated(self):
        expander = ExpandAmbiguities(_two_position_fixture(), max_trees=2)
        trees = list(expander)
        self.assertEqual(len(trees), 2)
        self.assertEqual(expander.count, 2)
        self.assertTrue(expander.truncated)
        self.assertEqual(trees, _four_expected()[:2])

    def test_max_trees_zero_yields_nothing_but_reports_truncated(self):
        expander = ExpandAmbiguities(_two_position_fixture(), max_trees=0)
        trees = list(expander)
        self.assertEqual(trees, [])
        self.assertTrue(expander.truncated)

    def test_max_trees_above_total_is_not_truncated(self):
        expander = ExpandAmbiguities(_two_position_fixture(), max_trees=10)
        trees = list(expander)
        self.assertEqual(len(trees), 4)
        self.assertFalse(expander.truncated)

    def test_exact_cap_is_not_truncated(self):
        expander = ExpandAmbiguities(_two_position_fixture(), max_trees=4)
        trees = list(expander)
        self.assertEqual(len(trees), 4)
        self.assertFalse(expander.truncated)

    def test_truncated_is_set_even_if_closed_right_after_cap(self):
        expander = ExpandAmbiguities(_two_position_fixture(), max_trees=2)
        it = iter(expander)
        next(it)
        next(it)
        it.close()  # close immediately after the cap-th tree
        self.assertEqual(expander.count, 2)
        self.assertTrue(expander.truncated)

    def test_modifying_one_result_does_not_affect_others(self):
        trees = list(ExpandAmbiguities(_two_position_fixture()))
        trees[0].data = 'CHANGED'
        trees[0].children.append(Tree('extra', ['x']))
        trees[0].children[0].data = 'CHANGED_A'
        trees[0].children[0].children.append('999')
        # The other three results must be untouched.
        self.assertEqual(trees[1], _four_expected()[1])
        self.assertEqual(trees[2], _four_expected()[2])
        self.assertEqual(trees[3], _four_expected()[3])

    def test_each_result_has_independent_child_lists(self):
        trees = list(ExpandAmbiguities(_two_position_fixture()))
        self.assertIsNot(trees[0].children, trees[1].children)
        self.assertIsNot(trees[0].children[0], trees[1].children[0])

    def test_token_metadata_preserved_and_independent(self):
        tok = Token('NUM', '42', start_pos=5, line=2, column=3,
                    end_line=2, end_column=5, end_pos=7)
        tree = Tree('root', [Tree('_ambig', [Tree('x', [tok]), Tree('x', [tok])])])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 2)
        t0 = trees[0].children[0].children[0]
        t1 = trees[1].children[0].children[0]
        self.assertIsNot(t0, t1)
        self.assertEqual((t0.start_pos, t0.line, t0.column), (5, 2, 3))
        self.assertEqual((t0.end_line, t0.end_column, t0.end_pos), (2, 5, 7))
        t0.value = '99'
        t0.type = 'CHANGED'
        self.assertEqual(t1.value, '42')
        self.assertEqual(t1.type, 'NUM')

    def test_tree_meta_preserved(self):
        meta = Meta()
        meta.line = 7
        meta.column = 3
        tree = Tree('root', [Tree('_ambig', [Tree('a', ['1'], meta=meta), Tree('a', ['2'])])])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 2)
        self.assertIsNot(trees[0].children[0].meta, meta)
        self.assertEqual(trees[0].children[0].meta.line, 7)
        self.assertEqual(trees[0].children[0].meta.column, 3)

    def test_shared_subtree_expanded_in_both_positions(self):
        # The same Tree object is reachable from two positions. It must be
        # expanded independently in both, not mistaken for a cycle.
        shared = Tree('shared', [Tree('_ambig', [Tree('s', ['1']), Tree('s', ['2'])])])
        tree = Tree('root', [shared, shared])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 4)
        self.assertEqual(trees, [
            Tree('root', [Tree('shared', [Tree('s', ['1'])]), Tree('shared', [Tree('s', ['1'])])]),
            Tree('root', [Tree('shared', [Tree('s', ['1'])]), Tree('shared', [Tree('s', ['2'])])]),
            Tree('root', [Tree('shared', [Tree('s', ['2'])]), Tree('shared', [Tree('s', ['1'])])]),
            Tree('root', [Tree('shared', [Tree('s', ['2'])]), Tree('shared', [Tree('s', ['2'])])]),
        ])

    def test_shared_token_expanded_in_both_positions(self):
        tok = Token('T', 'x')
        tree = Tree('root', [Tree('_ambig', [Tree('a', [tok]), Tree('a', [tok])])])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 2)
        self.assertIsNot(trees[0].children[0].children[0], trees[1].children[0].children[0])

    def test_true_self_loop_is_rejected(self):
        loop = Tree('loop', [])
        loop.children.append(loop)  # self-cycle
        tree = Tree('root', [Tree('_ambig', [Tree('ok', ['1']), loop])])
        trees = list(ExpandAmbiguities(tree))  # must not hang
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0], Tree('root', [Tree('ok', ['1'])]))

    def test_true_mutual_cycle_is_rejected(self):
        a = Tree('A', [])
        b = Tree('B', [a])
        a.children.append(b)  # A -> B -> A
        tree = Tree('root', [Tree('_ambig', [Tree('ok', ['x']), a])])
        trees = list(ExpandAmbiguities(tree))  # must not hang
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0], Tree('root', [Tree('ok', ['x'])]))

    def test_nested_ambig_nodes(self):
        # An _ambig alternative that itself contains an _ambig node.
        inner = Tree('_ambig', [Tree('i', ['1']), Tree('i', ['2'])])
        tree = Tree('root', [Tree('_ambig', [inner, Tree('o', ['3'])])])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 3)
        self.assertEqual(trees, [
            Tree('root', [Tree('i', ['1'])]),
            Tree('root', [Tree('i', ['2'])]),
            Tree('root', [Tree('o', ['3'])]),
        ])

    def test_tree_without_ambig_yields_itself(self):
        tree = Tree('root', [Tree('a', ['1']), Token('T', 'x')])
        trees = list(ExpandAmbiguities(tree))
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0], tree)

    def test_early_close_stops_expansion(self):
        tree = _two_position_fixture()
        expander = ExpandAmbiguities(tree)
        it = iter(expander)
        next(it)
        it.close()  # close before the cap; must stop, no background expansion
        self.assertEqual(expander.count, 1)
        self.assertFalse(expander.truncated)

    def test_first_result_is_not_the_only_representative(self):
        # The first tree is just one combination; all must be produced.
        trees = list(ExpandAmbiguities(_two_position_fixture()))
        self.assertEqual(len(trees), 4)
        self.assertNotEqual(trees[0], trees[1])
        self.assertNotEqual(trees[0], trees[2])
        self.assertNotEqual(trees[0], trees[3])


if __name__ == '__main__':
    unittest.main()
