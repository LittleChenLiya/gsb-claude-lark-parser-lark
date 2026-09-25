"""Demo: error-recovery suggestions based on the LALR InteractiveParser.

Given a failing input, RepairSuggester searches (within a bounded cost and
state budget) for the cheapest token insertions/deletions that let the parse
finish, then verifies each suggestion by replaying it on a copy of the
failed parser state.  User callbacks (e.g. an embedded Transformer) do not
run during the search; they run only when the chosen repair is applied.

Run with:  python examples/error_recovery_demo.py
"""

import time

from lark import Lark, Token, UnexpectedToken
from lark.error_recovery import RepairSuggester

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


def demo(parser, suggester, text):
    print('=' * 60)
    print('Input: %r' % text)
    try:
        parser.parse(text)
    except UnexpectedToken as e:
        print('Parse failed at line %s, column %s (token %r)'
              % (e.line, e.column, e.token))
        started = time.perf_counter()
        result = suggester.suggest_from_error(e, INSERT_CANDIDATES)
        elapsed = time.perf_counter() - started
        print('Search: %s, %d states explored in %.3fs'
              % (result.status.value, result.states_explored, elapsed))
        if not result.repairs:
            print('No repair found within the cost/state budget.')
            return
        for i, repair in enumerate(result.repairs, 1):
            actions = ', '.join(str(a) for a in repair.actions)
            print('Repair #%d (cost %s): %s' % (i, repair.cost, actions))
        chosen = result.repairs[0]
        tree = chosen.apply(e.interactive_parser)
        print('Replay of repair #1 succeeded:')
        print(tree.pretty(), end='')
    else:
        print('Parsed successfully, no recovery needed.')


def main():
    parser = Lark(GRAMMAR, parser='lalr')
    suggester = RepairSuggester(max_cost=3, max_states=2000)
    demo(parser, suggester, '1+(2')      # missing right parenthesis
    demo(parser, suggester, '1 + + 2')   # extra operator
    demo(parser, suggester, '1 + 2')     # valid input


if __name__ == '__main__':
    main()
