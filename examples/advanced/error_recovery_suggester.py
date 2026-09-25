"""
Error-Recovery Suggester Demo
=============================

Demonstrates lark.error_recovery: given a failed LALR parse, search for the
cheapest token insertions/deletions that repair it, then replay the chosen
fix. Prints the error location, the repair sequence, and the replay result.

Run with:  python examples/advanced/error_recovery_suggester.py
"""

from copy import copy

from lark import Lark, Token, UnexpectedToken
from lark.error_recovery import suggest_error_recovery

GRAMMAR = r"""
?start: expr
?expr: term (PLUS term)*
?term: factor (MUL factor)*
?factor: NUMBER | LPAR expr RPAR
NUMBER: /\d+/
PLUS: "+"
MUL: "*"
LPAR: "("
RPAR: ")"
%ignore " "
"""

# Whitelist of tokens the search is allowed to insert. Token instances
# double as templates: their value is used for the inserted token.
INSERT_CANDIDATES = [
    Token('NUMBER', '0'),
    Token('PLUS', '+'),
    Token('MUL', '*'),
    Token('RPAR', ')'),
]


def remaining_tokens(error):
    """The failing token plus everything still left in the lexer."""
    ip = error.interactive_parser
    rest = list(copy(ip.lexer_thread).lex(ip.parser_state))
    return [error.token] + rest


def demo(parser, text):
    print("Input: %r" % text)
    try:
        parser.parse(text)
        print("  parses fine, nothing to do")
        return
    except UnexpectedToken as e:
        token = e.token
        where = 'end of input' if token.type == '$END' else 'token %r' % token
        print("  Error: unexpected %s at line %s, column %s" % (where, token.line, token.column))
        ip = e.interactive_parser

        report = suggest_error_recovery(
            ip,
            remaining_tokens(e),
            insert_candidates=INSERT_CANDIDATES,
            max_suggestions=3,
        )
        print("  Search: status=%s, states explored=%d" % (report.status, report.states_explored))
        if not report.ok:
            print("  No repair found within the cost bound")
            return

        for i, suggestion in enumerate(report.suggestions):
            actions = ", ".join(map(str, suggestion.actions))
            print("  Suggestion %d (cost %s): %s" % (i + 1, suggestion.cost, actions))

        tree = report.best.replay(ip)
        print("  Replay of best suggestion succeeded. Tree:")
        for line in tree.pretty().splitlines():
            print("    " + line)


def main():
    parser = Lark(GRAMMAR, parser='lalr')
    demo(parser, "1 + (2 * 3")   # missing right parenthesis -> insert RPAR
    print()
    demo(parser, "1 + + 2")      # extra operator            -> delete PLUS
    print()
    demo(parser, "1 2")          # missing operator          -> several equal-cost fixes


if __name__ == '__main__':
    main()
