"""Error-recovery suggestions for LALR parsing, based on ``InteractiveParser``.

Given a failed parse state and the remaining tokens, :class:`RepairSuggester`
searches for the cheapest sequence of token insertions and deletions that
lets the parse finish successfully.

Search bounds
-------------
The search is a uniform-cost (Dijkstra-like) exploration over copies of the
parser state.  It is bounded by three knobs:

- ``max_cost``: candidate repairs whose total cost exceeds this are pruned.
- ``max_states``: at most this many search nodes are expanded.  When the
  budget is exhausted before the search space is, the result is marked
  ``SearchStatus.INCOMPLETE`` -- the returned repairs are the best found so
  far, and the absence of repairs does **not** prove that none exist.
- ``max_repairs``: at most this many repairs are collected.  Only repairs at
  the minimal cost are returned.

Side effects
------------
All probing happens on *copies* of the parser state, with the user callbacks
(e.g. an embedded ``Transformer``) disabled, so searching never mutates the
original parser and never triggers user code.  User callbacks run only when a
chosen repair is applied via :meth:`Repair.apply`, which replays exactly the
selected token sequence on a fresh copy of the interactive parser.
"""

from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappush, heappop
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .exceptions import UnexpectedToken
from .lexer import Token
from .parsers.lalr_interactive_parser import InteractiveParser


class RepairKind(Enum):
    """The kind of a single repair action."""
    INSERT = 'insert'
    DELETE = 'delete'


class SearchStatus(Enum):
    """Whether the search space was fully explored within the budget.

    ``COMPLETE``: the search space (within ``max_cost``) was exhausted, so the
        returned repairs are optimal and, if empty, no repair within
        ``max_cost`` exists.
    ``INCOMPLETE``: the state budget (``max_states``) ran out first.  The
        returned repairs are the best found so far; an empty result does not
        prove that no repair exists.
    """
    COMPLETE = 'complete'
    INCOMPLETE = 'incomplete'


@dataclass(frozen=True)
class RepairAction:
    """A single edit step of a repair.

    Attributes:
        kind: ``RepairKind.INSERT`` or ``RepairKind.DELETE``.
        token: For ``INSERT``, the token to feed to the parser.  For
            ``DELETE``, the token from the remaining stream to skip.
        position: Index into the remaining-token stream at which the action
            applies (the number of remaining tokens consumed before it).
    """
    kind: RepairKind
    token: Token
    position: int

    def __repr__(self):
        return '%s(%r@%d)' % (self.kind.value, self.token, self.position)


@dataclass
class Repair:
    """A verified repair: a sequence of edits that makes the parse succeed.

    Attributes:
        actions: The edit steps, in order of application.
        cost: Total cost of the repair (sum of insert/delete costs).
        tokens: The repaired token stream suffix -- exactly the tokens that
            must be fed to the parser (inserted tokens plus the remaining
            tokens that were not deleted) to finish the parse.
        verified: True if the repair passed an actual replay on a copy of
            the failed parser state.
    """
    actions: Tuple[RepairAction, ...]
    cost: float
    tokens: Tuple[Token, ...] = field(default=())
    verified: bool = False

    def apply(self, interactive_parser: InteractiveParser) -> Any:
        """Replay this repair on a copy of ``interactive_parser`` and return the result.

        This is the only operation that runs user callbacks (e.g. an embedded
        ``Transformer``): they fire exactly once, for this repair's token
        sequence.  The given ``interactive_parser`` is not modified.
        """
        cursor = interactive_parser.copy()
        for token in self.tokens:
            cursor.feed_token(token)
        last_token = self.tokens[-1] if self.tokens else None
        return cursor.feed_eof(last_token)


@dataclass
class RepairSearchResult:
    """The outcome of :meth:`RepairSuggester.suggest`.

    Attributes:
        status: ``SearchStatus.COMPLETE`` if the search space was exhausted
            within the budget, ``SearchStatus.INCOMPLETE`` otherwise.
        repairs: The cheapest repairs found, sorted by ``(cost, token types,
            positions)`` for a stable, deterministic order.  Only repairs at
            the minimal cost are included.
        states_explored: How many search nodes were expanded.
    """
    status: SearchStatus
    repairs: List[Repair]
    states_explored: int

    @property
    def complete(self) -> bool:
        """True iff the search exhausted its space within the budget."""
        return self.status is SearchStatus.COMPLETE

    def __bool__(self):
        return bool(self.repairs)


def _signature(actions: Tuple[RepairAction, ...]) -> Tuple[Tuple[str, int], ...]:
    """A stable tie-break key for equal-cost repairs: by token type, then position."""
    return tuple((a.token.type, a.position) for a in actions)


class RepairSuggester:
    """Search for the cheapest token insertions/deletions that repair a failed LALR parse.

    Parameters:
        insert_cost: Cost of inserting one token.
        delete_cost: Cost of deleting (skipping) one token.
        max_cost: Prune candidate repairs more expensive than this.
        max_states: Budget of search-node expansions.  If it runs out, the
            result is ``SearchStatus.INCOMPLETE``.
        max_repairs: Maximum number of repairs to collect.
    """

    def __init__(self, insert_cost: float = 1.0, delete_cost: float = 1.0,
                 max_cost: float = 3.0, max_states: int = 2000, max_repairs: int = 3):
        if insert_cost < 0 or delete_cost < 0:
            raise ValueError('Costs must be non-negative')
        if max_states <= 0:
            raise ValueError('max_states must be positive')
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.max_cost = max_cost
        self.max_states = max_states
        self.max_repairs = max_repairs

    def suggest(self, interactive_parser: InteractiveParser,
                remaining_tokens: Iterable[Token],
                insert_candidates: Iterable[Union[str, Token]]) -> RepairSearchResult:
        """Search for repairs from a failed parse state.

        Parameters:
            interactive_parser: The parser state at the point of failure
                (e.g. ``UnexpectedToken.interactive_parser``).  Not modified.
            remaining_tokens: The tokens that were not yet consumed, starting
                with the token the parser choked on.  A trailing ``$END``
                token is ignored.
            insert_candidates: Whitelist of tokens that may be inserted.
                Each item is either a token-type name (a ``Token`` with an
                empty value is fabricated) or a ``Token`` instance (used
                verbatim, with position information borrowed from the
                insertion point).

        Returns a :class:`RepairSearchResult`.  Every returned repair has been
        verified by an actual replay on a copy of the parser state (with user
        callbacks disabled, so the search has no observable side effects).
        """
        remaining = list(remaining_tokens)
        while remaining and remaining[-1].type == '$END':
            remaining.pop()
        candidates = self._normalize_candidates(insert_candidates)

        # Probe on a copy of the failed state, with user callbacks disabled so
        # that searching cannot trigger transformer/callback side effects.
        root = interactive_parser.copy(deepcopy_values=False)
        conf_no_callbacks = copy(root.parser_state.parse_conf)
        conf_no_callbacks.callbacks = {}
        root.parser_state.parse_conf = conf_no_callbacks

        seq = count()
        # Heap entries: (cost, tie-break signature, seq, (state, index, actions, fed_tokens))
        heap: List[Tuple[float, Tuple, int, Tuple]] = [
            (0.0, (), next(seq), (root, 0, (), ()))
        ]
        visited: Dict[Tuple[Tuple, int], float] = {}
        repairs: List[Repair] = []
        best_cost: Optional[float] = None
        explored = 0
        exhausted_budget = False
        seen_streams = set()

        while heap:
            cost, sig, _, (state, index, actions, fed) = heappop(heap)
            if best_cost is not None and cost > best_cost:
                break  # uniform-cost: nothing cheaper can show up later
            if explored >= self.max_states:
                exhausted_budget = True
                break
            explored += 1

            key = (tuple(state.parser_state.state_stack), index)
            # A node visited at a strictly lower cost dominates this one.
            # Equal-cost nodes are still expanded, so that distinct
            # same-cost repairs can be discovered.
            if visited.get(key, float('inf')) < cost:
                continue
            visited[key] = cost

            if index == len(remaining):
                # All remaining tokens consumed: does the parser accept EOF?
                probe = state.copy(deepcopy_values=False)
                try:
                    probe.feed_eof(remaining[-1] if remaining else None)
                except UnexpectedToken:
                    pass
                else:
                    if best_cost is None:
                        best_cost = cost
                    # Deduplicate repairs that result in the same token stream
                    stream_key = tuple((t.type, t.value) for t in fed)
                    if stream_key not in seen_streams and len(repairs) < self.max_repairs:
                        seen_streams.add(stream_key)
                        repairs.append(Repair(actions, cost, fed))
                    # Fall through: further insertions may yield same-cost repairs

            # 1. Match: feed the next remaining token as-is (free).
            if index < len(remaining):
                token = remaining[index]
                probe = state.copy(deepcopy_values=False)
                try:
                    probe.feed_token(token)
                except UnexpectedToken:
                    pass
                else:
                    heappush(heap, (cost, sig, next(seq),
                                    (probe, index + 1, actions, fed + (token,))))

            # 2. Delete: skip the next remaining token.
            if index < len(remaining):
                new_cost = cost + self.delete_cost
                if new_cost <= self.max_cost and (best_cost is None or new_cost <= best_cost):
                    action = RepairAction(RepairKind.DELETE, remaining[index], index)
                    new_actions = actions + (action,)
                    heappush(heap, (new_cost, sig + _signature(new_actions[-1:]), next(seq),
                                    (state, index + 1, new_actions, fed)))

            # 3. Insert: feed a whitelisted token that the current state accepts.
            if candidates:
                choices = state.choices()
                for cand in candidates:
                    if cand.type not in choices:
                        continue
                    new_cost = cost + self.insert_cost
                    if new_cost > self.max_cost or (best_cost is not None and new_cost > best_cost):
                        continue
                    anchor = remaining[index] if index < len(remaining) else (remaining[-1] if remaining else None)
                    if anchor is not None:
                        token = Token.new_borrow_pos(cand.type, cand.value, anchor)
                    else:
                        token = Token(cand.type, cand.value)
                    probe = state.copy(deepcopy_values=False)
                    try:
                        probe.feed_token(token)
                    except UnexpectedToken:
                        continue
                    action = RepairAction(RepairKind.INSERT, token, index)
                    new_actions = actions + (action,)
                    heappush(heap, (new_cost, sig + _signature(new_actions[-1:]), next(seq),
                                    (probe, index, new_actions, fed + (token,))))

        status = SearchStatus.INCOMPLETE if exhausted_budget else SearchStatus.COMPLETE
        repairs = [r for r in repairs if self._verify(interactive_parser, r)]
        repairs.sort(key=lambda r: (r.cost, _signature(r.actions)))
        return RepairSearchResult(status, repairs, explored)

    def suggest_from_error(self, error: UnexpectedToken,
                           insert_candidates: Iterable[Union[str, Token]]) -> RepairSearchResult:
        """Like :meth:`suggest`, but starts from a caught ``UnexpectedToken``.

        The remaining tokens are taken from the exception: the offending
        token plus everything left in the lexer.  If the parse failed at
        end-of-input, the remaining stream is empty.
        """
        interactive_parser = error.interactive_parser
        if interactive_parser is None:
            raise ValueError('The exception does not carry an interactive parser')
        if error.token.type == '$END':
            remaining: List[Token] = []
        else:
            lexer_thread = copy(interactive_parser.lexer_thread)
            rest = list(lexer_thread.lex(interactive_parser.parser_state))
            remaining = [error.token] + rest
        return self.suggest(interactive_parser, remaining, insert_candidates)

    @staticmethod
    def _normalize_candidates(candidates: Iterable[Union[str, Token]]) -> List[Token]:
        normalized = []
        seen = set()
        for cand in candidates:
            token = cand if isinstance(cand, Token) else Token(str(cand), '')
            if token.type not in seen:
                seen.add(token.type)
                normalized.append(token)
        return normalized

    def _verify(self, interactive_parser: InteractiveParser, repair: Repair) -> bool:
        """Replay the repair on a fresh copy of the failed state (no user callbacks).

        Only repairs that actually parse to completion are reported.
        """
        cursor = interactive_parser.copy(deepcopy_values=False)
        conf_no_callbacks = copy(cursor.parser_state.parse_conf)
        conf_no_callbacks.callbacks = {}
        cursor.parser_state.parse_conf = conf_no_callbacks
        try:
            for token in repair.tokens:
                cursor.feed_token(token)
            cursor.feed_eof(repair.tokens[-1] if repair.tokens else None)
        except UnexpectedToken:
            return False
        repair.verified = True
        return True
