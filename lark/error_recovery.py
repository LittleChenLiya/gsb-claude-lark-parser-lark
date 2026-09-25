"""Error-recovery suggestions for LALR parsing, based on InteractiveParser.

Given a failed parse (an UnexpectedToken exception and its interactive
parser) plus the remaining tokens, suggest_error_recovery() searches for
the cheapest sequence of token insertions and deletions that makes the
parse succeed.

The search is performed on copies of the parser state, with all user
callbacks (tree-building / transformer / lexer callbacks) disabled, so
searching never mutates the original parser and never triggers user side
effects. Only the final, explicit RepairSuggestion.replay() runs the
chosen repair path for real, exactly once, with user callbacks enabled.
"""

from copy import copy
from dataclasses import dataclass
import heapq
from itertools import count
from typing import List, Optional, Sequence, Tuple, Union

from .exceptions import UnexpectedToken
from .lexer import Token
from .parsers.lalr_interactive_parser import InteractiveParser
from .parsers.lalr_parser_state import ParserState, ParseConf


#: Token types the caller allows the search to insert (a whitelist).
CandidateSpec = Union[str, Token]


@dataclass(frozen=True)
class RepairAction:
    """A single edit step of a repair sequence.

    Attributes:
        kind: Either 'insert' or 'delete'.
        token_type: The type of the inserted or deleted token.
        position: Index into the remaining-token stream the action applies to.
            An insertion at position 'i' happens right before the 'i'-th
            remaining token (len(remaining) means "at end of input").
        token: The concrete token that was inserted or deleted.
    """
    kind: str
    token_type: str
    position: int
    token: Token = None  # type: ignore[assignment]

    def __str__(self) -> str:
        return "%s %s at position %d" % (self.kind, self.token_type, self.position)


class RepairSuggestion:
    """One verified repair sequence, found by suggest_error_recovery().

    Attributes:
        actions: The edit steps, in application order.
        cost: Total cost of the sequence (sum of insert/delete costs).
        tokens: The repaired token stream (without the trailing '$END').
    """

    def __init__(self, actions: Sequence[RepairAction], cost: float, tokens: Sequence[Token]):
        self.actions: Tuple[RepairAction, ...] = tuple(actions)
        self.cost = cost
        self.tokens: Tuple[Token, ...] = tuple(tokens)

    def replay(self, interactive_parser: InteractiveParser):
        """Replay this repair on a copy of the interactive parser and return the result.

        This is the only place user callbacks (transformers, tree builders,
        lexer callbacks) run: exactly once, for this single chosen path.
        The passed-in interactive parser is left untouched.
        """
        cursor = interactive_parser.copy()
        for token in self.tokens:
            cursor.feed_token(token)
        last_token = self.tokens[-1] if self.tokens else None
        return cursor.feed_eof(last_token)

    def _sort_key(self):
        return (self.cost, tuple((a.token_type, a.position, a.kind) for a in self.actions))

    def __repr__(self) -> str:
        return "RepairSuggestion(cost=%r, actions=[%s])" % (self.cost, ", ".join(map(str, self.actions)))


class RecoveryReport:
    """The result of suggest_error_recovery().

    Attributes:
        status: 'complete' if the search space was fully explored within
            the cost bound and state budget (an empty suggestions list
            then genuinely means "no repair exists within the cost bound").
            'incomplete' if the state budget ran out first - in that case
            more (possibly cheaper) repairs may exist, and the absence of
            suggestions must not be read as "no solution".
        suggestions: Verified repair sequences, cheapest first. Suggestions
            of equal cost are ordered deterministically by token type and
            position.
        states_explored: How many search states were expanded.
    """

    COMPLETE = 'complete'
    INCOMPLETE = 'incomplete'

    def __init__(self, status: str, suggestions: List[RepairSuggestion], states_explored: int):
        assert status in (self.COMPLETE, self.INCOMPLETE)
        self.status = status
        self.suggestions = suggestions
        self.states_explored = states_explored

    @property
    def ok(self) -> bool:
        """Whether at least one verified repair was found."""
        return bool(self.suggestions)

    @property
    def best(self) -> Optional[RepairSuggestion]:
        """The cheapest verified suggestion, or None."""
        return self.suggestions[0] if self.suggestions else None

    def __repr__(self) -> str:
        return "RecoveryReport(status=%r, suggestions=%r, states_explored=%r)" % (
            self.status, self.suggestions, self.states_explored)


def _silent_parse_conf(parse_conf: ParseConf) -> ParseConf:
    """A copy of the parse configuration with all user callbacks removed."""
    conf = copy(parse_conf)
    conf.callbacks = {}
    return conf


def _copy_state(state: ParserState) -> ParserState:
    """Copy a parser state, sharing the (unused) lexer thread."""
    return ParserState(state.parse_conf, state.lexer,
                       list(state.state_stack), list(state.value_stack))


def _try_feed(state: ParserState, token: Token, is_end: bool = False) -> Optional[ParserState]:
    """Feed token to a copy of state; return the new state, or None if rejected."""
    new_state = _copy_state(state)
    try:
        new_state.feed_token(token, is_end)
    except UnexpectedToken:
        return None
    return new_state


def _silent_cursor(interactive_parser: InteractiveParser) -> InteractiveParser:
    """A callback-free copy of the interactive parser, for trial parses."""
    cursor = interactive_parser.copy(deepcopy_values=False)
    cursor.parser_state.parse_conf = _silent_parse_conf(cursor.parser_state.parse_conf)
    return cursor


def _synthesize_token(token_type: str, value: str, borrow_from: Optional[Token]) -> Token:
    if borrow_from is not None:
        return Token.new_borrow_pos(token_type, value, borrow_from)
    return Token(token_type, value, 0, 1, 1)


def _apply_actions(remaining: Sequence[Token], actions: Sequence[RepairAction]) -> List[Token]:
    """Compute the repaired token stream by applying insert/delete actions."""
    inserts = {}
    deleted = set()
    for action in actions:
        if action.kind == 'insert':
            inserts.setdefault(action.position, []).append(action.token)
        else:
            deleted.add(action.position)
    out: List[Token] = []
    for i, token in enumerate(remaining):
        out.extend(inserts.get(i, ()))
        if i not in deleted:
            out.append(token)
    out.extend(inserts.get(len(remaining), ()))
    return out


def _verify(interactive_parser: InteractiveParser, tokens: Sequence[Token]) -> bool:
    """Check that the repaired token stream actually parses, without user callbacks."""
    cursor = _silent_cursor(interactive_parser)
    try:
        for token in tokens:
            cursor.feed_token(token)
        cursor.feed_eof(tokens[-1] if tokens else None)
    except UnexpectedToken:
        return False
    return True


def _lex_remaining(interactive_parser: InteractiveParser) -> List[Token]:
    """Lex the rest of the input from a copy of the lexer thread (non-destructive)."""
    lexer_thread = copy(interactive_parser.lexer_thread)
    return list(lexer_thread.lex(interactive_parser.parser_state))


def suggest_error_recovery(interactive_parser: InteractiveParser,
                           remaining_tokens: Optional[Sequence[Token]] = None,
                           insert_candidates: Sequence[CandidateSpec] = (),
                           insert_cost: float = 1.0,
                           delete_cost: float = 1.0,
                           max_cost: float = 4.0,
                           state_budget: int = 10000,
                           max_suggestions: int = 1) -> RecoveryReport:
    """Search for the cheapest token insertions/deletions that repair a failed parse.

    The search is a uniform-cost (Dijkstra) search over parser states. It runs
    entirely on copies of the parser state with user callbacks disabled, so
    the original parser is never modified and no transformer/lexer side
    effects fire during the search. Every returned suggestion is verified by
    an actual (callback-free) replay before it is reported.

    Parameters:
        interactive_parser: The interactive parser at the point of failure,
            e.g. UnexpectedToken.interactive_parser. Not modified.
        remaining_tokens: The tokens that were not yet consumed, starting
            with the token that caused the error (typically
            [e.token] + rest). A trailing '$END' token is treated as
            the end-of-input marker. If None, the rest of the input is
            lexed from a copy of the lexer thread.
        insert_candidates: Whitelist of token types (names or Token
            instances) that the search is allowed to insert. When a Token
            instance is given, its value is used for the inserted token
            (its position information is ignored). Deletions are always
            allowed. Defaults to no insertions.
        insert_cost: Cost of inserting one token.
        delete_cost: Cost of deleting one token.
        max_cost: Cost bound; partial repairs more expensive than this are
            pruned. Together with state_budget this keeps the search finite.
        state_budget: Maximum number of search states to expand. If the
            budget runs out before the search space is exhausted, the report
            status is 'incomplete' (not "no solution").
        max_suggestions: Maximum number of verified suggestions to return.

    Returns:
        A RecoveryReport. Suggestions are ordered by cost; equal-cost
        suggestions are ordered deterministically by token type and position.
    """
    if remaining_tokens is None:
        remaining_tokens = _lex_remaining(interactive_parser)
    remaining = list(remaining_tokens)

    # Split off a trailing end-of-input marker; the goal test feeds it itself.
    eof_source: Optional[Token] = None
    if remaining and remaining[-1].type == '$END':
        eof_source = remaining.pop()
    elif remaining:
        eof_source = remaining[-1]
    else:
        eof_source = interactive_parser.lexer_thread.state.last_token

    # Insertion whitelist: token type -> value used for synthesized tokens.
    templates = {}
    for c in insert_candidates:
        if isinstance(c, Token):
            templates.setdefault(c.type, str(c))
        else:
            templates.setdefault(c, '')
    candidate_types = sorted(templates)

    silent_conf = _silent_parse_conf(interactive_parser.parser_state.parse_conf)
    start_state = ParserState(silent_conf,
                              interactive_parser.parser_state.lexer,
                              list(interactive_parser.parser_state.state_stack),
                              list(interactive_parser.parser_state.value_stack))

    eof_token = _synthesize_token('$END', '', eof_source)

    # Search nodes: (parser_state, index into remaining, actions so far)
    tiebreak = count()
    heap: List[Tuple[float, int, Tuple[ParserState, int, Tuple[RepairAction, ...]]]] = []
    best_seen = {}

    def push(cost: float, state: ParserState, index: int, actions: Tuple[RepairAction, ...]) -> None:
        key = (tuple(state.state_stack), index)
        if best_seen.get(key, float('inf')) <= cost:
            return
        best_seen[key] = cost
        heapq.heappush(heap, (cost, next(tiebreak), (state, index, actions)))

    push(0.0, start_state, 0, ())

    suggestions: List[RepairSuggestion] = []
    explored = 0
    status = RecoveryReport.COMPLETE

    while heap and len(suggestions) < max_suggestions:
        cost, _, (state, index, actions) = heapq.heappop(heap)
        explored += 1
        if explored > state_budget:
            status = RecoveryReport.INCOMPLETE
            break

        if index == len(remaining):
            # All remaining tokens consumed: is the parse complete?
            if _try_feed(state, eof_token, is_end=True) is not None:
                tokens = _apply_actions(remaining, actions)
                if _verify(interactive_parser, tokens):
                    suggestions.append(RepairSuggestion(actions, cost, tokens))
                continue

        if index < len(remaining):
            token = remaining[index]
            # 1. Match: consume the next token as-is (free).
            matched = _try_feed(state, token)
            if matched is not None:
                push(cost, matched, index + 1, actions)
            # 2. Delete: skip the next token.
            new_cost = cost + delete_cost
            if new_cost <= max_cost:
                push(new_cost, state, index + 1,
                     actions + (RepairAction('delete', token.type, index, token),))

        # 3. Insert: feed a whitelisted token without consuming input.
        new_cost = cost + insert_cost
        if new_cost <= max_cost:
            borrow_from = remaining[index] if index < len(remaining) else eof_source
            for token_type in candidate_types:
                inserted = _synthesize_token(token_type, templates[token_type], borrow_from)
                new_state = _try_feed(state, inserted)
                if new_state is not None:
                    push(new_cost, new_state, index,
                         actions + (RepairAction('insert', token_type, index, inserted),))

    suggestions.sort(key=RepairSuggestion._sort_key)
    return RecoveryReport(status, suggestions, explored)


__all__ = (
    'RepairAction',
    'RepairSuggestion',
    'RecoveryReport',
    'suggest_error_recovery',
)
