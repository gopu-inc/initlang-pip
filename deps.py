#!/usr/bin/env python3
"""
INITLANG - Langage de programmation innovant
Syntaxe: let x ==> 5, init.ger("Hello")
"""

import sys
import os
import argparse
from enum import Enum
from typing import List, Optional, Dict, Any

# ==================== VERSION ====================

__version__ = "copyright (c) gopu-inc componement as | 1.0.0"
__author__ = "Mauricio"

# ==================== LEXER ====================

class TokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    LET = "LET"
    FI = "FI"
    ARROW = "ARROW"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    INIT_GER = "INIT_GER"
    EOF = "EOF"

class Token:
    def __init__(self, type: TokenType, value: str = "", line: int = 1, column: int = 1):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type.value}, '{self.value}', {self.line}:{self.column})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if source else ''
    
    def advance(self):
        if self.position < len(self.source) - 1:
            self.position += 1
            self.current_char = self.source[self.position]
            self.column += 1
        else:
            self.current_char = ''
    
    def peek(self, n: int = 1) -> str:
        pos = self.position + n
        return self.source[pos] if pos < len(self.source) else ''
    
    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            if self.current_char == '\n':
                self.line += 1
                self.column = 1
            self.advance()
    
    def skip_comment(self):
        # CORRECTION : Support des commentaires avec #
        if self.current_char == '#':
            while self.current_char and self.current_char != '\n':
                self.advance()
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_column = self.column
        result = []
        
        while (self.current_char and 
               (self.current_char.isalnum() or self.current_char in ['_', '.'])):
            result.append(self.current_char)
            self.advance()
        
        identifier = ''.join(result)
        
        if identifier == "init.ger":
            return Token(TokenType.INIT_GER, identifier, start_line, start_column)
        
        keywords = {
            "let": TokenType.LET,
            "fi": TokenType.FI,
        }
        
        token_type = keywords.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, start_line, start_column)
    
    def read_number(self) -> Token:
        start_line = self.line
        start_column = self.column
        result = []
        
        # CORRECTION : Permettre les nombres qui commencent par des chiffres
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            result.append(self.current_char)
            self.advance()
        
        return Token(TokenType.NUMBER, ''.join(result), start_line, start_column)
    
    def read_string(self) -> Token:
        start_line = self.line
        start_column = self.column
        quote = self.current_char
        result = []
        
        self.advance()
        
        while self.current_char and self.current_char != quote:
            result.append(self.current_char)
            self.advance()
        
        if self.current_char != quote:
            raise SyntaxError(f"Unterminated string at line {self.line}")
        
        self.advance()
        return Token(TokenType.STRING, ''.join(result), start_line, start_column)
    
    def next_token(self) -> Token:
        # CORRECTION : Skip comments and whitespace
        while self.current_char and (self.current_char.isspace() or self.current_char == '#'):
            if self.current_char.isspace():
                self.skip_whitespace()
            elif self.current_char == '#':
                self.skip_comment()
        
        if not self.current_char:
            return Token(TokenType.EOF, "", self.line, self.column)
        
        # Identifiants (doit être avant les nombres)
        if self.current_char.isalpha() or self.current_char == '_':
            return self.read_identifier()
        
        # Nombres
        if self.current_char.isdigit():
            return self.read_number()
        
        # Chaînes de caractères
        if self.current_char in ['"', "'"]:
            return self.read_string()
        
        current_char = self.current_char
        current_line = self.line
        current_column = self.column
        
        # Opérateur arrow ==>
        if (current_char == '=' and self.peek() == '=' and self.peek(2) == '>'):
            self.advance()
            self.advance()
            self.advance()
            return Token(TokenType.ARROW, "==>", current_line, current_column)
        
        # Opérateurs simples
        operators = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ',': TokenType.COMMA,
        }
        
        if current_char in operators:
            self.advance()
            return Token(operators[current_char], current_char, current_line, current_column)
        
        raise SyntaxError(f"Unexpected character '{current_char}' at line {current_line}:{current_column}")
    
    def tokenize(self) -> List[Token]:
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens

# ==================== AST ====================

class ASTNode:
    pass

class Expression(ASTNode):
    pass

class NumberLiteral(Expression):
    def __init__(self, value: float):
        self.value = value

class StringLiteral(Expression):
    def __init__(self, value: str):
        self.value = value

class Identifier(Expression):
    def __init__(self, name: str):
        self.name = name

class BinaryExpression(Expression):
    def __init__(self, operator: TokenType, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right

class CallExpression(Expression):
    def __init__(self, function: Expression, arguments: List[Expression]):
        self.function = function
        self.arguments = arguments

class Statement(ASTNode):
    pass

class ExpressionStatement(Statement):
    def __init__(self, expression: Expression):
        self.expression = expression

class VariableDeclaration(Statement):
    def __init__(self, name: str, value: Expression):
        self.name = name
        self.value = value

class Program(ASTNode):
    def __init__(self, statements: Optional[List[Statement]] = None):
        self.statements = statements or []

# ==================== PARSER ====================

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.next_token()
        self.peek_token = self.lexer.next_token()
    
    def next_token(self):
        self.current_token = self.peek_token
        self.peek_token = self.lexer.next_token()
    
    def expect_peek(self, token_type: TokenType) -> bool:
        if self.peek_token.type == token_type:
            self.next_token()
            return True
        return False
    
    def error(self, message: str):
        raise SyntaxError(f"{message} at line {self.current_token.line}:{self.current_token.column}")
    
    def parse_program(self) -> Program:
        program = Program()
        while self.current_token.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                program.statements.append(stmt)
            self.next_token()
        return program
    
    def parse_statement(self) -> Optional[Statement]:
        if self.current_token.type == TokenType.LET:
            return self.parse_let_statement()
        else:
            return self.parse_expression_statement()
    
    def parse_let_statement(self) -> VariableDeclaration:
        self.next_token()  # skip 'let'
        
        # CORRECTION : Permettre les identifiants qui commencent par des chiffres
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected identifier after 'let'")
        
        name = self.current_token.value
        self.next_token()  # skip identifier
        
        if self.current_token.type != TokenType.ARROW:
            self.error("Expected '==>' after variable name")
        
        self.next_token()  # skip '==>'
        value = self.parse_expression(self.LOWEST)
        
        return VariableDeclaration(name, value)
    
    def parse_expression_statement(self) -> ExpressionStatement:
        expr = self.parse_expression(self.LOWEST)
        return ExpressionStatement(expr)
    
    # Précedence
    LOWEST = 1
    SUM = 2
    PRODUCT = 3
    
    PRECEDENCES = {
        TokenType.PLUS: SUM,
        TokenType.MINUS: SUM,
        TokenType.SLASH: PRODUCT,
        TokenType.STAR: PRODUCT,
    }
    
    def current_precedence(self) -> int:
        return self.PRECEDENCES.get(self.current_token.type, self.LOWEST)
    
    def peek_precedence(self) -> int:
        return self.PRECEDENCES.get(self.peek_token.type, self.LOWEST)
    
    def parse_expression(self, precedence: int) -> Optional[Expression]:
        left = self.parse_prefix()
        if not left:
            return None
        
        while (self.peek_token.type != TokenType.EOF and 
               precedence < self.peek_precedence()):
            left = self.parse_infix(left)
            if not left:
                return None
        
        return left
    
    def parse_prefix(self) -> Optional[Expression]:
        token = self.current_token
        
        if token.type == TokenType.IDENTIFIER:
            return self.parse_identifier()
        elif token.type == TokenType.NUMBER:
            return self.parse_number_literal()
        elif token.type == TokenType.STRING:
            return self.parse_string_literal()
        elif token.type == TokenType.INIT_GER:
            return self.parse_init_ger()
        elif token.type == TokenType.LPAREN:
            return self.parse_grouped_expression()
        else:
            return None
    
    def parse_infix(self, left: Expression) -> Optional[Expression]:
        if self.current_token.type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]:
            return self.parse_binary_expression(left)
        elif self.current_token.type == TokenType.LPAREN:
            return self.parse_call_expression(left)
        else:
            return left
    
    def parse_identifier(self) -> Identifier:
        return Identifier(self.current_token.value)
    
    def parse_number_literal(self) -> NumberLiteral:
        try:
            value = float(self.current_token.value)
            return NumberLiteral(value)
        except ValueError:
            return NumberLiteral(0)
    
    def parse_string_literal(self) -> StringLiteral:
        return StringLiteral(self.current_token.value)
    
    def parse_init_ger(self) -> Expression:
        self.next_token()  # skip 'init.ger'
        
        if self.current_token.type != TokenType.LPAREN:
            self.error("Expected '(' after init.ger")
        
        self.next_token()  # skip '('
        arg = self.parse_expression(self.LOWEST)
        
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after init.ger argument")
        
        return CallExpression(Identifier("init_ger"), [arg])
    
    def parse_grouped_expression(self) -> Optional[Expression]:
        self.next_token()  # skip '('
        expr = self.parse_expression(self.LOWEST)
        
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after expression")
        
        return expr
    
    def parse_binary_expression(self, left: Expression) -> BinaryExpression:
        op = self.current_token.type
        precedence = self.current_precedence()
        
        self.next_token()  # skip operator
        right = self.parse_expression(precedence)
        
        if not right:
            right = NumberLiteral(0)
        
        return BinaryExpression(op, left, right)
    
    def parse_call_expression(self, function: Expression) -> CallExpression:
        args = self.parse_call_arguments()
        return CallExpression(function, args)
    
    def parse_call_arguments(self) -> List[Expression]:
        args = []
        
        if self.peek_token.type == TokenType.RPAREN:
            self.next_token()
            return args
        
        self.next_token()
        arg = self.parse_expression(self.LOWEST)
        if arg:
            args.append(arg)
        
        while self.peek_token.type == TokenType.COMMA:
            self.next_token()
            self.next_token()
            arg = self.parse_expression(self.LOWEST)
            if arg:
                args.append(arg)
        
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after arguments")
        
        return args

# ==================== INTERPRETER ====================

class Environment:
    def __init__(self):
        self.store: Dict[str, Any] = {}
    
    def get(self, name: str) -> Any:
        if name in self.store:
            return self.store[name]
        else:
            raise NameError(f"Variable '{name}' not defined")
    
    def set(self, name: str, value: Any):
        self.store[name] = value

class Interpreter:
    def __init__(self):
        self.environment = Environment()
        self.environment.set("init_ger", lambda x: print(x))
    
    def interpret(self, program: Program):
        for statement in program.statements:
            self.evaluate_statement(statement)
    
    def evaluate_statement(self, stmt: Statement):
        if isinstance(stmt, ExpressionStatement):
            return self.evaluate_expression(stmt.expression)
        elif isinstance(stmt, VariableDeclaration):
            value = self.evaluate_expression(stmt.value)
            self.environment.set(stmt.name, value)
            return value
    
    def evaluate_expression(self, expr: Expression) -> Any:
        if isinstance(expr, NumberLiteral):
            return expr.value
        elif isinstance(expr, StringLiteral):
            return expr.value
        elif isinstance(expr, Identifier):
            return self.environment.get(expr.name)
        elif isinstance(expr, BinaryExpression):
            left = self.evaluate_expression(expr.left)
            right = self.evaluate_expression(expr.right)
            
            if expr.operator == TokenType.PLUS:
                return str(left) + str(right) if isinstance(left, str) or isinstance(right, str) else left + right
            elif expr.operator == TokenType.MINUS:
                return left - right
            elif expr.operator == TokenType.STAR:
                return left * right
            elif expr.operator == TokenType.SLASH:
                return left / right if right != 0 else float('inf')
        elif isinstance(expr, CallExpression):
            if isinstance(expr.function, Identifier):
                func = self.environment.get(expr.function.name)
                if callable(func):
                    args = [self.evaluate_expression(arg) for arg in expr.arguments]
                    return func(*args)
        return None

# ==================== CLI ====================

def show_help():
    print(f"""
INITLANG {__version__}

Usage:
  initlang [OPTIONS] [FILE]

Options:
  -h, --help          Show this help message
  -v, --version       Show version information
  -c, --command CODE  Execute code directly

Examples:
  initlang                    # Start REPL
  initlang script.init        # Execute file
  initlang -c "let x ==> 5"  # Execute command

Syntax:
  let x ==> 5
  init.ger("Hello World!")
  # This is a comment
    """)

def show_version():
    print(f"INITLANG version {__version__}")

def execute_code(code: str):
    try:
        lexer = Lexer(code)
        parser = Parser(lexer)
        program = parser.parse_program()
        interpreter = Interpreter()
        interpreter.interpret(program)
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) == 1:
        # REPL Mode
        print(f"=== INITLANG {__version__} ===")
        print("Type 'exit' to quit, 'help' for help")
        
        while True:
            try:
                line = input("INITLANG> ").strip()
                
                if line.lower() in ['exit', 'quit']:
                    break
                elif line.lower() == 'help':
                    show_help()
                    continue
                elif not line:
                    continue
                
                execute_code(line)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
    
    elif sys.argv[1] in ['-h', '--help']:
        show_help()
    
    elif sys.argv[1] in ['-v', '--version']:
        show_version()
    
    elif sys.argv[1] in ['-c', '--command'] and len(sys.argv) > 2:
        execute_code(sys.argv[2])
    
    elif len(sys.argv) == 2:
        # File execution
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            return
        
        with open(filename, 'r') as f:
            code = f.read()
        
        execute_code(code)
    
    else:
        show_help()

if __name__ == "__main__":
    main()
