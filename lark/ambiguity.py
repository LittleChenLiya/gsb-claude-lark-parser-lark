"""Explicit expansion of ambiguous parse forests.

When the Earley parser is used with ``ambiguity='explicit'``, ambiguous
derivations are returned wrapped in ``_ambig`` tree nodes (i.e. a forest).
This module provides :class:`AmbiguityExpander`, an iterator that lazily
expands such a forest into the individual parse trees - one tree at a time,
in a fixed, deterministic order - instead of materializing every possible
combination up front, and instead of silently keeping only the first parse.

Example:
    ::

        parser = Lark(grammar, parser='earley', ambiguity='explicit')
        forest = parser.parse(text)

        expander = AmbiguityExpander(forest, max_expansions=100)
        for tree in expander:
            print(tree.pretty())
        if expander.truncated:
            print('... more parses exist')
"""

from typing import Dict, Iterator, Optional, Set

from .exceptions import LarkError
from .tree import Tree

__all__ = ('AmbiguityExpander', 'AmbiguityCycleError', 'expand_ambiguities')


class AmbiguityCycleError(LarkError):
    """Raised when an ``_ambig`` forest cannot be expanded because it
    contains a genuine cycle (a node that is its own ancestor).

    Subtrees that are merely *shared* between several branches (a DAG) are
    perfectly fine and are never mistaken for a cycle.
    """


def _count_expansions(node, in_path, counts):
    # type: (object, Set[int], Dict[int, int]) -> int
    """Return the number of distinct expansions of ``node``.

    ``in_path`` holds the ids of the ``Tree`` nodes on the *current* DFS
    path, so only genuine cycles are rejected.  ``counts`` memoizes the
    result for every fully-visited node, which means shared subtrees are
    counted once and never misdetected as cycles (a global "visited" set
    alone would not be able to tell the two cases apart).
    """
    if not isinstance(node, Tree):
        # Tokens and other leaves expand to exactly themselves.
        return 1
    nid = id(node)
    if nid in counts:
        return counts[nid]
    if nid in in_path:
        raise AmbiguityCycleError(
            "Cannot expand an '_ambig' forest that contains a cycle "
            "(a %r node is its own ancestor)" % (node.data,))
    in_path.add(nid)
    try:
        if node.data == '_ambig':
            # Each child is an alternative derivation.
            total = 0
            for child in node.children:
                total += _count_expansions(child, in_path, counts)
        else:
            # Cartesian product of the children's expansions.
            total = 1
            for child in node.children:
                total *= _count_expansions(child, in_path, counts)
    finally:
        in_path.discard(nid)
    counts[nid] = total
    return total


def _node_count(node, counts):
    # type: (object, Dict[int, int]) -> int
    if isinstance(node, Tree):
        return counts[id(node)]
    return 1


def _expand_kth(node, k, counts):
    # type: (object, int, Dict[int, int]) -> object
    """Build the ``k``-th expansion of ``node`` (0-based).

    Every call creates brand new ``Tree`` nodes (preserving ``data`` and
    ``meta``) with a brand new children list, so the returned trees are
    structurally independent from one another.  Tokens and other leaves are
    kept as-is, preserving their metadata.
    """
    if not isinstance(node, Tree):
        return node
    if node.data == '_ambig':
        # Pick the alternative that contains the k-th expansion, in the
        # original order of the children.
        for child in node.children:
            n = _node_count(child, counts)
            if k < n:
                return _expand_kth(child, k, counts)
            k -= n
        raise IndexError("expansion index %d out of range" % k)
    # A regular node: expand each child by mixed-radix decomposition of k,
    # so that the last child varies fastest (standard Cartesian order,
    # following the original order of the children).
    m = len(node.children)
    indexes = [0] * m
    rem = k
    for i in range(m - 1, -1, -1):
        n = _node_count(node.children[i], counts)
        indexes[i] = rem % n
        rem //= n
    new_children = [_expand_kth(child, idx, counts)
                    for child, idx in zip(node.children, indexes)]
    return type(node)(node.data, new_children, meta=node._meta)


class AmbiguityExpander(Iterator[Tree]):
    """Lazily iterate over the individual parse trees of an ``_ambig`` forest.

    The expansions are produced one at a time, in a fixed order: the
    children of each ``_ambig`` node are tried in their original order, and
    the Cartesian combinations of sibling positions are enumerated with the
    last position varying fastest.

    Parameters:
        forest: A ``Tree`` that may contain ``_ambig`` nodes, as returned by
            the Earley parser with ``ambiguity='explicit'``.
        max_expansions: Optional cap on how many trees will be produced.
            When the forest contains more expansions than the cap, iteration
            stops after ``max_expansions`` trees and ``truncated`` is True.

    Attributes:
        total: Total number of expansions in the forest (computed up-front
            in O(size of forest); may be a huge integer).
        limit: How many trees this iterator will actually produce
            (``min(total, max_expansions)``).
        truncated: True if the cap prevented some expansions from being
            produced.
        count: How many trees have been produced so far.

    The iterator never uses the first parse as a stand-in for the rest: all
    expansions are enumerated (up to the cap), and truncation is always
    reported through ``truncated``.  Iteration may be stopped at any moment
    with :meth:`close` (or by using the expander as a context manager);
    no further expansions are computed afterwards.

    Raises:
        AmbiguityCycleError: if the forest contains a genuine cycle.
            Shared subtrees (a DAG) are supported and are not cycles.
    """

    def __init__(self, forest, max_expansions=None):
        # type: (Tree, Optional[int]) -> None
        if max_expansions is not None and max_expansions < 0:
            raise ValueError("max_expansions must be >= 0, got %r" % (max_expansions,))
        self._forest = forest
        self._counts = {}  # type: Dict[int, int]
        self.total = _count_expansions(forest, set(), self._counts)  # type: int
        self.max_expansions = max_expansions
        self.limit = self.total if max_expansions is None else min(self.total, max_expansions)  # type: int
        self.truncated = self.limit < self.total  # type: bool
        self.count = 0  # type: int
        self._closed = False

    def __iter__(self):
        # type: () -> AmbiguityExpander
        return self

    def __next__(self):
        # type: () -> Tree
        if self._closed or self.count >= self.limit:
            raise StopIteration
        result = _expand_kth(self._forest, self.count, self._counts)
        self.count += 1
        return result

    def __len__(self):
        # type: () -> int
        """The number of trees this iterator will produce (``limit``)."""
        return self.limit

    def close(self):
        # type: () -> None
        """Stop the iteration early. No further expansions are computed."""
        self._closed = True
        self._forest = None
        self._counts = {}

    def __enter__(self):
        # type: () -> AmbiguityExpander
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # type: (object, object, object) -> None
        self.close()


def expand_ambiguities(forest, max_expansions=None):
    # type: (Tree, Optional[int]) -> AmbiguityExpander
    """Return an :class:`AmbiguityExpander` over the given ``_ambig`` forest.

    Convenience wrapper; see :class:`AmbiguityExpander` for details.
    """
    return AmbiguityExpander(forest, max_expansions=max_expansions)
