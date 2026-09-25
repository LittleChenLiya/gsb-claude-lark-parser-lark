# Error Recovery

Lark can suggest how to repair input that fails to parse, using the LALR
`InteractiveParser`. The `lark.error_recovery` module searches for the
cheapest sequence of token insertions and deletions that lets the parse
finish successfully.

## Quick start

```python
from lark import Lark, Token, UnexpectedToken
from lark.error_recovery import RepairSuggester

parser = Lark(grammar, parser='lalr')
suggester = RepairSuggester()

try:
    parser.parse(text)
except UnexpectedToken as e:
    result = suggester.suggest_from_error(
        e, insert_candidates=[Token('RPAR', ')'), Token('NUMBER', '0')])
    if result.repairs:
        repair = result.repairs[0]
        print(repair.actions)          # the edit sequence
        tree = repair.apply(e.interactive_parser)   # replay it
```

## How it works

`RepairSuggester.suggest()` takes the failed interactive parser, the
remaining tokens (starting with the token the parser choked on), and a
whitelist of insert candidates, and runs a uniform-cost search over copies
of the parser state:

- **Match** - feed the next remaining token as-is (free).
- **Delete** - skip the next remaining token (costs `delete_cost`).
- **Insert** - feed a whitelisted token the current state accepts
  (costs `insert_cost`).

A candidate repair succeeds when all remaining tokens are consumed and the
parser accepts end-of-input. Every returned repair is verified by an actual
replay of its token sequence on a fresh copy of the failed parser state.

Insert candidates are provided by the caller as a whitelist, either as
token-type names (`'RPAR'`) or as `Token` instances (`Token('RPAR', ')')`).
Use `Token` instances when the inserted token's value matters (it shows up
in the resulting tree).

## Search bounds

The search is bounded by three parameters of `RepairSuggester`:

- `max_cost` (default `3.0`) - candidate repairs more expensive than this
  are pruned.
- `max_states` (default `2000`) - the budget of search-node expansions.
- `max_repairs` (default `3`) - how many repairs to collect.

The result's `status` distinguishes two outcomes:

- `SearchStatus.COMPLETE` - the search space (within `max_cost`) was
  exhausted. The returned repairs are optimal, and an empty list proves no
  repair exists within `max_cost`.
- `SearchStatus.INCOMPLETE` - the `max_states` budget ran out first. The
  returned repairs are the best found so far, and an empty list does **not**
  prove that no repair exists.

Only repairs at the minimal cost are returned. Repairs of equal cost are
ordered deterministically: by token type, then by position in the remaining
token stream. Repairs that would produce the same token stream are
de-duplicated.

## Side effects

Searching probes on copies of the parser state with user callbacks disabled,
so it never mutates the original parser and never runs user code (such as an
embedded `Transformer`). User callbacks run only when a repair is applied
with `Repair.apply()`, which replays exactly the selected repair's token
sequence on a copy of the interactive parser and returns the parse result.

## API

- `RepairSuggester(insert_cost=1.0, delete_cost=1.0, max_cost=3.0, max_states=2000, max_repairs=3)`
  - `suggest(interactive_parser, remaining_tokens, insert_candidates)`
  - `suggest_from_error(error, insert_candidates)` - convenience wrapper
    around `UnexpectedToken`.
- `RepairSearchResult` - `status`, `repairs`, `states_explored`, and the
  `complete` property.
- `Repair` - `actions`, `cost`, `tokens`, `verified`, and `apply()`.
- `RepairAction` - `kind` (`RepairKind.INSERT`/`RepairKind.DELETE`),
  `token`, `position`.

See `examples/error_recovery_demo.py` for a runnable demonstration.
