"""Completion analysis for LALR parsers.

This module implements a completion-analysis tool on top of the LALR
``InteractiveParser``. Given a (possibly incomplete) input prefix, it advances
a parser over the prefix and reports, for the position at the end of the
prefix:

- the candidate token types (terminals) that may follow,
- the literal text of candidates whose terminal is a fixed string
  (e.g. keywords), when it can be determined directly,
- the grammar rule positions (rule + index in its expansion) that cause
  each candidate to appear.

The analysis is read-only: it never modifies, extends or auto-completes the
input, and it never mutates the ``Lark`` instance it was created from, so the
same parser (and any existing ``InteractiveParser``) keeps working normally
afterwards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .common import ParserConf
from .exceptions import ConfigurationError, UnexpectedInput
from .grammar import Rule
from .lexer import PatternStr, Token
from .parsers.grammar_analysis import State
from .parsers.lalr_analysis import LALR_Analyzer, ParseTable, Reduce
from .parsers.lalr_parser_state import ParseConf, ParserState


@dataclass(frozen=True)
class RulePosition:
    """A position inside a grammar rule that expects a candidate terminal.

    Attributes:
        rule: The grammar rule.
        index: The index in ``rule.expansion`` of the symbol that expects the
            candidate terminal (i.e. ``rule.expansion[index]`` is the terminal,
            or a non-terminal through which it is reached).
    """
    rule: Rule
    index: int

    @property
    def origin(self) -> str:
        "Name of the rule this position belongs to."
        return self.rule.origin.name

    @property
    def expected_symbol(self) -> Optional[str]:
        """Name of the symbol at this position.

        ``None`` when the position is at the end of the rule, i.e. the rule
        is complete (used for ``'$END'`` candidates).
        """
        if self.index < len(self.rule.expansion):
            return self.rule.expansion[self.index].name
        return None

    def __repr__(self):
        before = ' '.join(s.name for s in self.rule.expansion[:self.index])
        after = ' '.join(s.name for s in self.rule.expansion[self.index:])
        return '<%s : %s * %s>' % (self.rule.origin.name, before, after)


@dataclass(frozen=True)
class CompletionCandidate:
    """A single completion candidate.

    Attributes:
        token_type: The type (name) of the acceptable terminal. ``'$END'``
            means the input may legally end here (EOF is acceptable).
        literal: The exact text the terminal matches, when the terminal is
            defined as a fixed string (e.g. a keyword). ``None`` otherwise.
        positions: The grammar rule positions that expect this terminal.
    """
    token_type: str
    literal: Optional[str]
    positions: Tuple[RulePosition, ...] = field(default_factory=tuple)

    def __repr__(self):
        literal = ' %r' % self.literal if self.literal is not None else ''
        return '<Candidate %s%s>' % (self.token_type, literal)


@dataclass(frozen=True)
class CompletionResult:
    """The result of analyzing an input prefix.

    Attributes:
        candidates: The completion candidates at the end of the consumed prefix.
        tokens_consumed: The tokens that were successfully fed to the parser.
        error: The ``UnexpectedInput`` (e.g. ``UnexpectedToken`` /
            ``UnexpectedCharacters``) raised while advancing over the prefix,
            or ``None`` if the whole prefix was consumed. When an error
            occurred, the candidates describe the position at which it happened.
        input_exhausted: True if the entire input was consumed without error.
        accepts_eof: True if the input may legally end at this position.
    """
    candidates: Tuple[CompletionCandidate, ...]
    tokens_consumed: Tuple[Token, ...]
    error: Optional[UnexpectedInput]
    input_exhausted: bool
    accepts_eof: bool

    @property
    def token_types(self) -> List[str]:
        "The token types of all candidates, in order."
        return [c.token_type for c in self.candidates]

    def candidate(self, token_type: str) -> Optional[CompletionCandidate]:
        "Return the candidate for ``token_type``, or None if not present."
        for c in self.candidates:
            if c.token_type == token_type:
                return c
        return None


class CompletionAnalyzer:
    """Analyzes input prefixes and reports completion candidates.

    Built on top of the LALR ``InteractiveParser``: each call to ``analyze()``
    creates a fresh interactive parser, advances it over the given prefix, and
    inspects the resulting parser position. The analyzer is stateless between
    calls, so it may be reused for consecutive analyses.

    Parameters:
        parser: A ``Lark`` instance using ``parser='lalr'``.
        start: The start rule to analyze for. Required only if the parser was
            created with multiple start rules.
    """

    def __init__(self, parser, start: Optional[str] = None):
        if parser.options.parser != 'lalr':
            raise ConfigurationError(
                "CompletionAnalyzer only works with parser='lalr' (got %r)" % parser.options.parser)

        start_decls = list(parser.options.start)
        if start is None:
            if len(start_decls) > 1:
                raise ConfigurationError(
                    "Lark initialized with more than 1 possible start rule. Must specify which start rule to analyze",
                    start_decls)
            start ,= start_decls
        elif start not in start_decls:
            raise ConfigurationError(
                "Unknown start rule %s. Must be one of %r" % (start, start_decls))

        self.lark = parser
        self.start = start

        # Terminals defined as plain strings have a directly determinable literal
        self._literals: Dict[str, str] = {
            t.name: t.pattern.value
            for t in parser.terminals
            if isinstance(t.pattern, PatternStr)
        }

        # Re-run the LALR analysis in debug mode, to get a parse table whose
        # states are the LR(0) item-sets (sets of rule positions). This lets us
        # attribute every candidate token to the rule positions expecting it,
        # without touching the parser's own (integer-keyed) table.
        parser_conf = ParserConf(list(parser.rules), {}, [start])
        analyzer = LALR_Analyzer(parser_conf, debug=True)
        analyzer.compute_lalr()
        parse_table = analyzer.parse_table
        assert isinstance(parse_table, ParseTable)
        self.parse_table: ParseTable = parse_table
        # No callbacks: we only track states, never build values
        self._parse_conf = ParseConf(parse_table, {}, start)

    def _positions_for(self, state: State, token_type: str) -> Tuple[RulePosition, ...]:
        """Rule positions in ``state`` that were reached by shifting ``token_type``."""
        positions = []
        for rp in state:
            if rp.index > 0:
                sym = rp.rule.expansion[rp.index - 1]
                if sym.is_term and sym.name == token_type:
                    positions.append(RulePosition(rp.rule, rp.index - 1))
        positions.sort(key=lambda p: (p.rule.origin.name, p.index, str(p.rule)))
        return tuple(positions)

    def _eof_positions_for(self, state: ParserState) -> Tuple[RulePosition, ...]:
        """Rule positions completed along the reduce chain that EOF would trigger."""
        # EOF never shifts; it triggers a chain of reductions. Walk that chain
        # (without callbacks) and collect every rule position it completes.
        trial = state.copy(deepcopy_values=False)
        states = self._parse_conf.states
        end_state = self._parse_conf.end_state
        positions = []
        seen = set()
        while True:
            position = trial.position
            if position in seen:
                break
            seen.add(position)
            for rp in position:
                if rp.is_satisfied and not rp.rule.origin.name.startswith('$root_'):
                    positions.append(RulePosition(rp.rule, rp.index))
            if position == end_state:
                break
            try:
                action, rule = states[position]['$END']
            except KeyError:
                break
            if action is not Reduce:
                break
            size = len(rule.expansion)
            if size:
                del trial.state_stack[-size:]
                del trial.value_stack[-size:]
            _action, new_state = states[trial.state_stack[-1]][rule.origin.name]
            trial.state_stack.append(new_state)
            trial.value_stack.append(None)
        positions = list(dict.fromkeys(positions))
        positions.sort(key=lambda p: (p.rule.origin.name, p.index, str(p.rule)))
        return tuple(positions)

    def analyze(self, text: str = '') -> CompletionResult:
        """Advance a fresh interactive parser over ``text`` and analyze the final position.

        Returns a ``CompletionResult``. Does not modify ``text``, the analyzer,
        or the underlying ``Lark`` instance.
        """
        ip = self.lark.parse_interactive(text, start=self.start)
        state = ParserState(self._parse_conf, ip.lexer_thread)

        tokens: List[Token] = []
        error: Optional[UnexpectedInput] = None
        exhausted = True

        token_iter = ip.iter_parse()
        while True:
            try:
                token = next(token_iter)
            except StopIteration:
                break
            except UnexpectedInput as e:
                # Lexer failed on the remaining input (e.g. half a keyword)
                error = e
                exhausted = False
                break
            try:
                state.feed_token(token, token.type == '$END')
            except UnexpectedInput as e:
                # Parser rejected this token; candidates describe the position before it
                error = e
                exhausted = False
                break
            tokens.append(token)

        candidates = []
        actions = self.parse_table.states[state.position]
        for token_type in actions:
            if not token_type.isupper():    # not a terminal
                continue
            trial = state.copy(deepcopy_values=False)
            try:
                if token_type == '$END':
                    if tokens:
                        eof_token = Token.new_borrow_pos('$END', '', tokens[-1])
                    else:
                        eof_token = Token('$END', '', 0, 1, 1)
                    trial.feed_token(eof_token, is_end=True)
                else:
                    trial.feed_token(Token(token_type, ''))
            except UnexpectedInput:
                # A reduce action on this lookahead doesn't lead to a valid shift
                continue
            if token_type == '$END':
                # EOF never shifts; report the rules completed by its reduce chain
                positions = self._eof_positions_for(state)
            else:
                positions = self._positions_for(trial.position, token_type)
            candidates.append(CompletionCandidate(
                token_type=token_type,
                literal=self._literals.get(token_type),
                positions=positions,
            ))

        candidates.sort(key=lambda c: (c.token_type == '$END', c.token_type))

        return CompletionResult(
            candidates=tuple(candidates),
            tokens_consumed=tuple(tokens),
            error=error,
            input_exhausted=exhausted,
            accepts_eof=any(c.token_type == '$END' for c in candidates),
        )
