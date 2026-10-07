"""Completion analysis for LALR parsers.

This module implements a completion-assistance tool, aimed at editor-like
integrations (IDEs, REPLs, query builders) where the input may still be
incomplete while the user is typing.

Given a (possibly partial) input prefix, the :class:`CompletionAnalyzer`
advances a LALR ``InteractiveParser`` over the prefix, and reports which
terminals may legally follow at the current position. For every candidate
it reports:

- the candidate token type (terminal name),
- the literal text, when it can be statically determined (i.e. the
  terminal is defined by a plain string, such as a keyword),
- the rule positions (a rule, and an offset in its expansion) that caused
  the candidate to appear.

The analyzer only *observes* the parser state. It never rewrites the input,
never feeds synthetic tokens into the user's parser, and never mutates an
``InteractiveParser`` that was handed to it, so parsing can always resume
normally afterwards.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .exceptions import ConfigurationError, UnexpectedCharacters, UnexpectedToken
from .grammar import NonTerminal, Rule
from .lexer import PatternStr, TerminalDef, Token
from .parsers.grammar_analysis import RulePtr
from .parsers.lalr_analysis import Reduce, Shift
from .parsers.lalr_interactive_parser import InteractiveParser
from .utils import classify


@dataclass(frozen=True)
class RulePosition:
    """A position inside a grammar rule, pointing at the place where a
    candidate terminal is expected.

    Attributes:
        rule: The grammar rule.
        index: Offset inside ``rule.expansion``. When it equals
            ``len(rule.expansion)``, the rule is fully matched at this
            position, and the candidate comes from its follow-set (a
            reduce action in the parse table).
    """
    rule: Rule
    index: int

    @property
    def rule_name(self) -> str:
        "Name of the rule (its origin nonterminal)."
        return self.rule.origin.name

    @property
    def expected_symbol(self) -> Optional[str]:
        "Name of the symbol expected at this position, or None if the rule is fully matched."
        if self.index < len(self.rule.expansion):
            return self.rule.expansion[self.index].name
        return None

    def __str__(self) -> str:
        before = [x.name for x in self.rule.expansion[:self.index]]
        after = [x.name for x in self.rule.expansion[self.index:]]
        return '<%s : %s * %s>' % (self.rule.origin.name, ' '.join(before), ' '.join(after))


@dataclass(frozen=True)
class CompletionCandidate:
    """A single completion candidate: a terminal that is accepted at the
    current parse position.

    Attributes:
        token_type: The terminal (token type) name, e.g. ``"SELECT"``.
        literal: The exact text the terminal matches, when statically known
            (i.e. the terminal is defined by a plain string pattern).
            ``None`` for regex-defined terminals and for ``$END``.
        positions: The rule positions that caused this candidate to appear.
            May contain several entries when multiple rules share the token.
    """
    token_type: str
    literal: Optional[str]
    positions: Tuple[RulePosition, ...] = ()


@dataclass
class CompletionResult:
    """The outcome of a completion analysis.

    Attributes:
        candidates: Terminals accepted at the current position.
        accepts_eof: Whether the input may legally end at this position
            (i.e. ``$END`` is among the candidates).
        parsed_text: The leading part of the input that was successfully
            lexed and parsed. ``None`` when the result was computed from an
            existing ``InteractiveParser`` instead of a text prefix.
        partial_token: Trailing text that could not be lexed (e.g. half a
            keyword). Candidates are filtered down to terminals whose
            literal this text is a prefix of (regex terminals are kept,
            since they cannot be ruled out statically).
        unexpected_token: The token that the parser rejected, when the
            prefix is well-lexed but contains a parse error. Candidates
            then describe what may appear *instead* of this token.
        interactive_parser: An ``InteractiveParser`` advanced over the
            valid prefix. Parsing may be resumed from it (e.g. with
            ``resume_parse()``); it is safe to keep using it.
    """
    candidates: Tuple[CompletionCandidate, ...]
    accepts_eof: bool
    parsed_text: Optional[str] = None
    partial_token: str = ''
    unexpected_token: Optional[Token] = None
    interactive_parser: Optional[InteractiveParser] = None

    @property
    def token_types(self) -> Set[str]:
        "The set of acceptable token types at the current position."
        return {c.token_type for c in self.candidates}

    @property
    def literals(self) -> Dict[str, str]:
        "Mapping of token type -> literal text, for candidates whose literal is known."
        return {c.token_type: c.literal for c in self.candidates if c.literal is not None}

    @property
    def ok(self) -> bool:
        "True when the whole input prefix was lexed and parsed without errors."
        return not self.partial_token and self.unexpected_token is None

    def get_candidate(self, token_type: str) -> Optional[CompletionCandidate]:
        "Return the candidate for the given token type, or None."
        for c in self.candidates:
            if c.token_type == token_type:
                return c
        return None


class CompletionAnalyzer:
    """Analyzes incomplete input and reports completion candidates.

    Wraps a ``Lark`` instance configured with ``parser='lalr'``, and uses
    its LALR ``InteractiveParser`` to advance over input prefixes.

    Parameters:
        parser: A ``Lark`` instance (must use the LALR parser).
        start: Start rule to analyze for. Required only when the grammar
            declares multiple start rules.
    """

    def __init__(self, parser, start: Optional[str] = None):
        frontend = parser.parser
        if frontend.parser_conf.parser_type != 'lalr':
            raise ConfigurationError("CompletionAnalyzer requires parser='lalr'")

        self._lark = parser
        self._frontend = frontend
        self.start = frontend._verify_start(start)
        self._rules = frontend.parser_conf.rules
        self._terminals = frontend.lexer_conf.terminals_by_name
        self._ignore_case = bool(frontend.lexer_conf.g_regex_flags & frontend.lexer_conf.re_module.IGNORECASE)
        # Cache of {state: frozenset[RulePtr]} per start rule
        self._item_sets: Dict[str, Dict[object, FrozenSet[RulePtr]]] = {}

    def analyze(self, text: str, start: Optional[str] = None) -> CompletionResult:
        """Advance over the given input prefix, and analyze the position it ends at.

        Feeds as much of ``text`` into a fresh ``InteractiveParser`` as
        possible, then computes the completion candidates for the resulting
        position. The input is never modified or auto-completed.

        Parameters:
            text: The input prefix. May be empty, may end mid-token
                (e.g. half a keyword), and may contain errors.
            start: Override the start rule for this call.
        """
        start = self._frontend._verify_start(start) if start is not None else self.start
        ip = self._lark.parse_interactive(text, start=start)

        partial_token = ''
        unexpected_token = None
        parsed_until = len(text)
        try:
            for token in ip.lexer_thread.lex(ip.parser_state):
                ip.feed_token(token)
        except UnexpectedCharacters as e:
            # The tail of the input doesn't lex (e.g. half a keyword)
            pos = e.pos_in_stream if e.pos_in_stream is not None else len(text)
            parsed_until = pos
            partial_token = text[pos:]
        except UnexpectedToken as e:
            # The token lexed fine, but the parser rejects it here
            unexpected_token = e.token
            pos = getattr(e.token, 'start_pos', None)
            if pos is None:
                pos = ip.lexer_thread.state.line_ctr.char_pos
            parsed_until = pos

        candidates, accepts_eof = self._candidates_at(ip, partial_token or None)
        return CompletionResult(
            candidates=candidates,
            accepts_eof=accepts_eof,
            parsed_text=text[:parsed_until],
            partial_token=partial_token,
            unexpected_token=unexpected_token,
            interactive_parser=ip,
        )

    def analyze_interactive(self, interactive_parser: InteractiveParser) -> CompletionResult:
        """Compute completion candidates for the current position of an
        existing ``InteractiveParser``, without modifying it.

        May be called any number of times; the interactive parser remains
        usable for normal parsing afterwards.
        """
        candidates, accepts_eof = self._candidates_at(interactive_parser)
        return CompletionResult(
            candidates=candidates,
            accepts_eof=accepts_eof,
            interactive_parser=interactive_parser,
        )

    # -- internals ------------------------------------------------------

    def _candidates_at(self, ip: InteractiveParser, partial_token: Optional[str] = None
                       ) -> Tuple[Tuple[CompletionCandidate, ...], bool]:
        """Compute the candidates for the current state of ``ip``. Read-only."""
        parser_state = ip.parser_state
        parse_conf = parser_state.parse_conf
        start = parse_conf.start

        item_sets = self._item_sets.get(start)
        if item_sets is None:
            item_sets = self._item_sets[start] = self._compute_item_sets(start, parse_conf.parse_table)

        accepts = set(ip.accepts())
        if partial_token:
            # Input cannot legally end in the middle of an unlexable token
            accepts.discard('$END')

        state = parser_state.position
        item_set = item_sets.get(state, frozenset())
        table_row = parse_conf.states[state]

        candidates = []
        for token_type in sorted(accepts):
            positions = self._positions_for(token_type, item_set, table_row)
            termdef = self._terminals.get(token_type)
            literal = self._literal_of(termdef)
            candidate = CompletionCandidate(token_type, literal, tuple(positions))
            if partial_token and not self._matches_partial(candidate, termdef, partial_token):
                continue
            candidates.append(candidate)

        return tuple(candidates), '$END' in accepts

    def _positions_for(self, token_type: str, item_set: FrozenSet[RulePtr], table_row) -> List[RulePosition]:
        """All rule positions in the current state that give rise to ``token_type``."""
        positions = []
        seen = set()

        # Positions where the terminal is the very next symbol of a rule
        for rp in item_set:
            if not rp.is_satisfied and rp.next.is_term and rp.next.name == token_type:
                key = (rp.rule, rp.index)
                if key not in seen:
                    seen.add(key)
                    positions.append(RulePosition(rp.rule, rp.index))

        # Positions of fully matched rules whose follow-set contains the terminal
        action = table_row.get(token_type)
        if action is not None and action[0] is Reduce:
            rule = action[1]
            key = (rule, len(rule.expansion))
            if key not in seen:
                seen.add(key)
                positions.append(RulePosition(rule, len(rule.expansion)))

        return positions

    @staticmethod
    def _literal_of(termdef: Optional[TerminalDef]) -> Optional[str]:
        """The literal text of a terminal, when statically determinable."""
        if termdef is not None and isinstance(termdef.pattern, PatternStr):
            return termdef.pattern.value
        return None

    def _matches_partial(self, candidate: CompletionCandidate, termdef: Optional[TerminalDef], partial: str) -> bool:
        """Whether a candidate is compatible with the trailing unlexable text."""
        literal = candidate.literal
        if literal is None:
            # Regex-defined terminals cannot be ruled out statically
            return True
        ignore_case = self._ignore_case or (termdef is not None and 'i' in termdef.pattern.flags)
        if ignore_case:
            return literal.lower().startswith(partial.lower())
        return literal.startswith(partial)

    def _compute_item_sets(self, start: str, parse_table) -> Dict[object, FrozenSet[RulePtr]]:
        """Map every reachable parse-table state to its set of LR(0) items
        (rule positions), by walking the parse table from the start state
        and computing closures/gotos over the grammar rules.
        """
        root_rule = Rule(NonTerminal('$root_' + start), [NonTerminal(start)])
        rules_by_origin = classify(self._rules + [root_rule], lambda r: r.origin.name)

        def closure(ptrs) -> FrozenSet[RulePtr]:
            result = set(ptrs)
            agenda = list(ptrs)
            while agenda:
                rp = agenda.pop()
                if rp.is_satisfied or rp.next.is_term:
                    continue
                for rule in rules_by_origin.get(rp.next.name, ()):
                    next_rp = RulePtr(rule, 0)
                    if next_rp not in result:
                        result.add(next_rp)
                        agenda.append(next_rp)
            return frozenset(result)

        states = parse_table.states
        item_sets: Dict[object, FrozenSet[RulePtr]] = {}
        start_state = parse_table.start_states[start]
        work = [(start_state, closure({RulePtr(root_rule, 0)}))]
        while work:
            state, items = work.pop()
            if state in item_sets:
                continue
            item_sets[state] = items
            for symbol_name, (action, arg) in states[state].items():
                if action is not Shift:
                    continue
                advanced = [RulePtr(rp.rule, rp.index + 1) for rp in items
                            if not rp.is_satisfied and rp.next.name == symbol_name]
                if advanced:
                    work.append((arg, closure(advanced)))
        return item_sets
