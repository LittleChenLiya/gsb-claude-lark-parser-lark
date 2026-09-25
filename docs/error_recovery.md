# Error Recovery

Lark can suggest how to repair input that fails to parse, using the
LALR `InteractiveParser`. The function `suggest_error_recovery()` searches
for the cheapest sequence of token insertions and deletions that makes the
parse succeed, verifies each suggestion by actually replaying it, and can
then replay the chosen repair for real.

```python
from lark import Lark, Token, UnexpectedToken
from lark.error_recovery import suggest_error_recovery

parser = Lark(grammar, parser='lalr')

try:
    parser.parse('1 + (2 * 3')
except UnexpectedToken as e:
    print('Error at line %d, column %d' % (e.token.line, e.token.column))
    report = suggest_error_recovery(
        e.interactive_parser,
        [e.token],                       # remaining tokens, starting at the error
        insert_candidates=[Token('RPAR', ')'), Token('PLUS', '+')],
    )
    if report.ok:
        print(report.best.actions)       # [insert RPAR at position 0]
        tree = report.best.replay(e.interactive_parser)
```

## How it works

- The search is a uniform-cost (Dijkstra) search over parser states. From
  each state it may consume the next remaining token (free), delete it
  (`delete_cost`), or insert a whitelisted token (`insert_cost`).
- Insertions are limited to the caller-provided `insert_candidates`
  whitelist (token type names, or `Token` instances whose value is reused).
  Deletions of remaining tokens are always allowed.
- The search runs entirely on copies of the parser state, with all user
  callbacks disabled. The original parser is never modified, and no
  transformer or lexer-callback side effects fire while searching.
- Every returned suggestion is verified by an actual (callback-free) replay.
  User callbacks run only in `RepairSuggestion.replay()`, exactly once, for
  the single chosen repair path.
- Suggestions are ordered by cost. Suggestions of equal cost are ordered
  deterministically by token type and position, so results are stable.

## Search bounds

The search space grows exponentially with the repair length, so it is kept
finite by two bounds:

- `max_cost` (default `4.0`): partial repairs more expensive than this are
  pruned. Lower it for snappier, shallower repairs; raise it to find
  repairs that need many edits.
- `state_budget` (default `10000`): the maximum number of search states to
  expand. When the budget runs out before the search space is exhausted,
  the returned report has `status == 'incomplete'`: more (possibly cheaper)
  repairs may exist, and an empty suggestion list must **not** be read as
  "no solution". When the search finishes within the budget, the status is
  `'complete'`, and only then does an empty suggestion list genuinely mean
  "no repair exists within the cost bound".

`max_suggestions` (default `1`) caps how many verified suggestions are
collected; raise it to enumerate equal-cost alternatives.

## API

- `suggest_error_recovery(interactive_parser, remaining_tokens=None, insert_candidates=(), insert_cost=1.0, delete_cost=1.0, max_cost=4.0, state_budget=10000, max_suggestions=1)`
  - `interactive_parser`: the interactive parser at the point of failure
    (e.g. `UnexpectedToken.interactive_parser`). It is not modified.
  - `remaining_tokens`: the unconsumed tokens, starting with the token that
    caused the error (typically `[e.token] + rest`). A trailing `$END` token
    is treated as the end-of-input marker. If `None`, the rest of the input
    is lexed from a copy of the lexer thread.
  - Returns a `RecoveryReport`.
- `RecoveryReport`: `status` (`'complete'` / `'incomplete'`), `suggestions`,
  `states_explored`, `ok`, `best`.
- `RepairSuggestion`: `actions`, `cost`, `tokens`, and
  `replay(interactive_parser)` which returns the parse result.
- `RepairAction`: `kind` (`'insert'` / `'delete'`), `token_type`,
  `position`, `token`.

See `examples/advanced/error_recovery_suggester.py` for a runnable demo.
