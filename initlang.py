#!/usr/bin/env python3
"""
INITLANG - Langage de programmation innovant
Syntaxe: let x ==> 5, fi add(a, b) { }, init.ger("Hello")
"""

import sys
import os
import readline
from enum import Enum
from typing import List, Optional, Dict, Any

# ==================== LEXER ====================

class TokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    LET = "LET"
    FI = "FI"
    CONST = "CONST"
    RETURN = "RETURN"
    ARROW = "ARROW"
    DOUBLE_ARROW = "DOUBLE_ARROW"
    DOT = "DOT"
    COLON = "COLON"
    COMMA = "COMMA"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    INIT_GER = "INIT_GER"
    INIT_LOG = "INIT_LOG"
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
        if identifier == "init.log":
            return Token(TokenType.INIT_LOG, identifier, start_line, start_column)
        
        keywords = {
            "let": TokenType.LET,
            "fi": TokenType.FI,
            "const": TokenType.CONST,
            "return": TokenType.RETURN,
        }
        
        token_type = keywords.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, start_line, start_column)
    
    def read_number(self) -> Token:
        start_line = self.line
        start_column = self.column
        result = []
        has_dot = False
        
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    break
                has_dot = True
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
            if self.current_char == '\\':
                self.advance()
                escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', "'": "'", '\\': '\\'}
                result.append(escape_chars.get(self.current_char, self.current_char))
            else:
                result.append(self.current_char)
            self.advance()
        
        if self.current_char != quote:
            raise SyntaxError(f"Unterminated string at line {self.line}")
        
        self.advance()
        return Token(TokenType.STRING, ''.join(result), start_line, start_column)
    
    def next_token(self) -> Token:
        self.skip_whitespace()
        
        if not self.current_char:
            return Token(TokenType.EOF, "", self.line, self.column)
        
        if self.current_char.isalpha() or self.current_char == '_':
            return self.read_identifier()
        
        if self.current_char.isdigit():
            return self.read_number()
        
        if self.current_char in ['"', "'"]:
            return self.read_string()
        
        current_char = self.current_char
        current_line = self.line
        current_column = self.column
        
        if (current_char == '=' and self.peek() == '=' and self.peek(2) == '>'):
            self.advance()
            self.advance()
            self.advance()
            return Token(TokenType.ARROW, "==>", current_line, current_column)
        
        if current_char == '=' and self.peek() == '>':
            self.advance()
            self.advance()
            return Token(TokenType.DOUBLE_ARROW, "=>", current_line, current_column)
        
        operators = {
            '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR, '/': TokenType.SLASH,
            '(': TokenType.LPAREN, ')': TokenType.RPAREN, '{': TokenType.LBRACE, '}': TokenType.RBRACE,
            ',': TokenType.COMMA, ':': TokenType.COLON, '.': TokenType.DOT,
        }
        
        if current_char in operators:
            self.advance()
            return Token(operators[current_char], current_char, current_line, current_column)
        
        if current_char == '=' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.EQ, "==", current_line, current_column)
        
        if current_char == '!' and self.peek() == '=':
            self.advance()
            self.advance()
            return Token(TokenType.NEQ, "!=", current_line, current_column)
        
        if current_char == '<':
            self.advance()
            return Token(TokenType.LT, "<", current_line, current_column)
        
        if current_char == '>':
            self.advance()
            return Token(TokenType.GT, ">", current_line, current_column)
        
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
    def __repr__(self):
        return f"Number({self.value})"

class StringLiteral(Expression):
    def __init__(self, value: str):
        self.value = value
    def __repr__(self):
        return f"String('{self.value}')"

class Identifier(Expression):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"Identifier({self.name})"

class BinaryExpression(Expression):
    def __init__(self, operator: TokenType, left: Expression, right: Expression):
        self.operator = operator
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Binary({self.left} {self.operator.value} {self.right})"

class CallExpression(Expression):
    def __init__(self, function: Expression, arguments: List[Expression]):
        self.function = function
        self.arguments = arguments
    def __repr__(self):
        args = ', '.join(str(arg) for arg in self.arguments)
        return f"Call({self.function}({args}))"

class Statement(ASTNode):
    pass

class ExpressionStatement(Statement):
    def __init__(self, expression: Expression):
        self.expression = expression
    def __repr__(self):
        return f"ExpressionStmt({self.expression})"

class VariableDeclaration(Statement):
    def __init__(self, name: str, value: Expression, is_const: bool = False):
        self.name = name
        self.value = value
        self.is_const = is_const
    def __repr__(self):
        const_str = "const " if self.is_const else ""
        return f"VarDecl({const_str}{self.name} = {self.value})"

class FunctionDeclaration(Statement):
    def __init__(self, name: str, parameters: List[str], body: 'BlockStatement'):
        self.name = name
        self.parameters = parameters
        self.body = body
    def __repr__(self):
        params = ', '.join(self.parameters)
        return f"FuncDecl({self.name}({params}) {{ ... }})"

class BlockStatement(Statement):
    def __init__(self, statements: Optional[List[Statement]] = None):
        self.statements = statements or []
    def __repr__(self):
        stmts = '; '.join(str(stmt) for stmt in self.statements)
        return f"Block({{ {stmts} }})"

class ReturnStatement(Statement):
    def __init__(self, value: Optional[Expression] = None):
        self.value = value
    def __repr__(self):
        return f"Return({self.value})"

class Program(ASTNode):
    def __init__(self, statements: Optional[List[Statement]] = None):
        self.statements = statements or []
    def __repr__(self):
        stmts = '\n'.join(str(stmt) for stmt in self.statements)
        return f"Program:\n{stmts}"

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
        elif self.current_token.type == TokenType.FI:
            return self.parse_function_statement()
        elif self.current_token.type == TokenType.RETURN:
            return self.parse_return_statement()
        else:
            return self.parse_expression_statement()
    
    def parse_let_statement(self) -> VariableDeclaration:
        self.next_token()
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected identifier after 'let'")
        name = self.current_token.value
        self.next_token()
        if not self.expect_peek(TokenType.ARROW):
            self.error("Expected '==>' after variable name")
        self.next_token()
        value = self.parse_expression(self.LOWEST)
        return VariableDeclaration(name, value)
    
    def parse_function_statement(self) -> FunctionDeclaration:
        self.next_token()
        if self.current_token.type != TokenType.IDENTIFIER:
            self.error("Expected function name after 'fi'")
        name = self.current_token.value
        self.next_token()
        if not self.expect_peek(TokenType.LPAREN):
            self.error("Expected '(' after function name")
        params = self.parse_function_parameters()
        if not self.expect_peek(TokenType.LBRACE):
            self.error("Expected '{' after function parameters")
        body = self.parse_block_statement()
        return FunctionDeclaration(name, params, body)
    
    def parse_function_parameters(self) -> List[str]:
        params = []
        if self.peek_token.type == TokenType.RPAREN:
            self.next_token()
            return params
        self.next_token()
        if self.current_token.type == TokenType.IDENTIFIER:
            params.append(self.current_token.value)
        else:
            self.error("Expected parameter name")
            return params
        while self.peek_token.type == TokenType.COMMA:
            self.next_token()
            self.next_token()
            if self.current_token.type == TokenType.IDENTIFIER:
                params.append(self.current_token.value)
            else:
                self.error("Expected parameter name after ','")
                break
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after parameters")
        return params
    
    def parse_block_statement(self) -> BlockStatement:
        block = BlockStatement()
        self.next_token()
        while (self.current_token.type != TokenType.RBRACE and 
               self.current_token.type != TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
            self.next_token()
        return block
    
    def parse_return_statement(self) -> ReturnStatement:
        self.next_token()
        value = self.parse_expression(self.LOWEST)
        return ReturnStatement(value)
    
    def parse_expression_statement(self) -> ExpressionStatement:
        expr = self.parse_expression(self.LOWEST)
        return ExpressionStatement(expr)
    
    LOWEST = 1
    EQUALS = 2
    LESSGREATER = 3
    SUM = 4
    PRODUCT = 5
    PREFIX = 6
    CALL = 7
    
    PRECEDENCES = {
        TokenType.EQ: EQUALS, TokenType.NEQ: EQUALS,
        TokenType.LT: LESSGREATER, TokenType.GT: LESSGREATER,
        TokenType.PLUS: SUM, TokenType.MINUS: SUM,
        TokenType.SLASH: PRODUCT, TokenType.STAR: PRODUCT,
        TokenType.LPAREN: CALL,
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
        elif token.type in [TokenType.MINUS, TokenType.PLUS]:
            return self.parse_prefix_expression()
        else:
            self.error(f"No prefix parse function for {token.type}")
            return None
    
    def parse_infix(self, left: Expression) -> Optional[Expression]:
        if self.current_token.type in [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT
        ]:
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
            self.error(f"Could not parse number: {self.current_token.value}")
            return NumberLiteral(0)
    
    def parse_string_literal(self) -> StringLiteral:
        return StringLiteral(self.current_token.value)
    
    def parse_init_ger(self) -> Expression:
        self.next_token()
        if not self.expect_peek(TokenType.LPAREN):
            self.error("Expected '(' after init.ger")
        self.next_token()
        arg = self.parse_expression(self.LOWEST)
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after init.ger argument")
        return arg
    
    def parse_grouped_expression(self) -> Optional[Expression]:
        self.next_token()
        expr = self.parse_expression(self.LOWEST)
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after expression")
        return expr
    
    def parse_prefix_expression(self) -> Optional[Expression]:
        op = self.current_token.type
        self.next_token()
        right = self.parse_expression(self.PREFIX)
        if not right:
            return None
        if op == TokenType.MINUS:
            zero = NumberLiteral(0)
            return BinaryExpression(TokenType.MINUS, zero, right)
        return right
    
    def parse_binary_expression(self, left: Expression) -> BinaryExpression:
        op = self.current_token.type
        precedence = self.current_precedence()
        self.next_token()
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
        args.append(self.parse_expression(self.LOWEST))
        while self.peek_token.type == TokenType.COMMA:
            self.next_token()
            self.next_token()
            args.append(self.parse_expression(self.LOWEST))
        if not self.expect_peek(TokenType.RPAREN):
            self.error("Expected ')' after arguments")
        return args

# ==================== INTERPRETER ====================

class Environment:
    def __init__(self, parent=None):
        self.store: Dict[str, Any] = {}
        self.parent = parent
    
    def get(self, name: str) -> Any:
        if name in self.store:
            return self.store[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"Variable '{name}' not defined")
    
    def set(self, name: str, value: Any):
        self.store[name] = value

class Interpreter:
    def __init__(self):
        self.environment = Environment()
    
    def interpret(self, program: Program):
        result = None
        for statement in program.statements:
            result = self.evaluate_statement(statement)
        return result
    
    def evaluate_statement(self, stmt: Statement) -> Any:
        if isinstance(stmt, ExpressionStatement):
            return self.evaluate_expression(stmt.expression)
        elif isinstance(stmt, VariableDeclaration):
            value = self.evaluate_expression(stmt.value)
            self.environment.set(stmt.name, value)
            return value
        elif isinstance(stmt, ReturnStatement):
            return self.evaluate_expression(stmt.value) if stmt.value else None
        else:
            raise NotImplementedError(f"Statement type {type(stmt)} not implemented")
    
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
                return left + right
            elif expr.operator == TokenType.MINUS:
                return left - right
            elif expr.operator == TokenType.STAR:
                return left * right
            elif expr.operator == TokenType.SLASH:
                return left / right
            elif expr.operator == TokenType.EQ:
                return left == right
            elif expr.operator == TokenType.NEQ:
                return left != right
            elif expr.operator == TokenType.LT:
                return left < right
            elif expr.operator == TokenType.GT:
                return left > right
            else:
                raise NotImplementedError(f"Operator {expr.operator} not implemented")
        else:
            raise NotImplementedError(f"Expression type {type(expr)} not implemented")

# ==================== REPL & CLI ====================

class REPL:
    def __init__(self):
        self.interpreter = Interpreter()
    
    def start(self):
        print("=== INITLANG REPL ===")
        print("Type 'exit' to quit")
        print("Example: let x ==> 5")
        print("Example: init.ger(\"Hello INITLANG!\")")
        
        while True:
            try:
                line = input("INITLANG> ").strip()
                
                if line.lower() in ['exit', 'quit', 'q']:
                    break
                if not line:
                    continue
                
                lexer = Lexer(line)
                parser = Parser(lexer)
                program = parser.parse_program()
                
                result = self.interpreter.interpret(program)
                if result is not None:
                    print(result)
                    
            except Exception as e:
                print(f"Error: {e}")
    
    def run_file(self, filename: str):
        try:
            with open(filename, 'r') as f:
                code = f.read()
            
            lexer = Lexer(code)
            parser = Parser(lexer)
            program = parser.parse_program()
            
            result = self.interpreter.interpret(program)
            if result is not None:
                print(result)
                
        except Exception as e:
            print(f"Error executing {filename}: {e}")

def main():
    if len(sys.argv) == 1:
        repl = REPL()
        repl.start()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            sys.exit(1)
        repl = REPL()
        repl.run_file(filename)
    else:
        print("Usage: initlang [file.init]")
        print("If no file provided, starts REPL mode")

# ==================== TESTS & EXAMPLES ====================

def run_tests():
    """Exécute des tests basiques"""
    print("=== TESTING INITLANG ===")
    
    # Test 1: Lexer
    print("\n1. Testing Lexer:")
    code = "let x ==> 5"
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    for token in tokens:
        print(f"  {token}")
    
    # Test 2: Parser
    print("\n2. Testing Parser:")
    code = """
    let x ==> 5
    let y ==> x + 10 * 2
    init.ger("Result: " + y)
    """
    lexer = Lexer(code)
    parser = Parser(lexer)
    program = parser.parse_program()
    print(f"  {program}")
    
    # Test 3: Interpreter
    print("\n3. Testing Interpreter:")
    code = """
    let x ==> 5
    let y ==> x * 3
    y - 2
    """
    lexer = Lexer(code)
    parser = Parser(lexer)
    program = parser.parse_program()
    interpreter = Interpreter()
    result = interpreter.interpret(program)
    print(f"  Result: {result}")
    
    print("\n=== ALL TESTS COMPLETED ===")

def create_example_file():
    """Crée un fichier exemple"""
    example_code = '''let name ==> "Mauricio"
let age ==> 25
let message ==> "Hello " + name + "! You are " + age + " years old."

init.ger(message)
init.ger("2 + 3 = " + (2 + 3))
init.ger("10 * 2 = " + (10 * 2))

let x ==> 5
let y ==> x * 3
let result ==> y + 2
init.ger("Final result: " + result)
'''
    with open("example.init", "w") as f:
        f.write(example_code)
    print("Example file 'example.init' created!")

# ==================== EXECUTION ====================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "example":
        create_example_file()
    else:
        main()
