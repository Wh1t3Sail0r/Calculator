# CONSTANTS
import math

DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
degree_mode = True


# ERRORS
class Error:
    def __init__(self, pos_start, pos_end, name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.name = name
        self.details = details

    def as_string(self):
        error = f'{self.name}: {self.details}\n'
        error += f'  File {self.pos_start.file_name}, line {self.pos_start.line + 1}'
        return error


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        error = self.generate_traceback()
        error += f'{self.name}: {self.details}'
        return error

    def generate_traceback(self):
        error = ''
        pos = self.pos_start
        context = self.context
        while context:
            error = f'  File {pos.file_name}, line {pos.line + 1}, in {context.name}\n' + error
            pos = context.parent_pos
            context = context.parent
        return 'Traceback:\n' + error


# RUNTIME RESULT
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def __repr__(self):
        if self.error:
            return f'{self.error}'
        else:
            return f'{self.value}'

    def register(self, result):
        if result.error: self.error = result.error
        return result.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


# POSITION
class Position:
    def __init__(self, index, line, col, file_name, content):
        self.index = index
        self.line = line
        self.col = col
        self.file_name = file_name
        self.content = content

    def advance(self, current_char=None):
        self.index += 1
        self.col += 1

        if current_char == '\n':
            self.line += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.index, self.line, self.col, self.file_name, self.content)


# TOKENS
TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_POW = 'POW'
TT_FACT = 'FACT'
TT_EOF = 'EOF'
TT_FUNC = 'FUNCTION'

FUNCTIONS = [
    'sin',
    'cos',
    'tan',
    'csc',
    'sec',
    'cot'
]

EQUATIONS = [
    'solve'
]

class Token:
    def __init__(self, type, value=None, pos_start=None, pos_end=None):
        self.type = type
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end.copy()

    def __repr__(self):
        if self.value: return f'{self.type}: {self.value}'
        return f'{self.type}'


# LEXER:
class Lexer:
    def __init__(self, file_name, text):
        self.file_name = file_name
        self.text = text
        self.pos = Position(-1, 0, -1, file_name, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None

    def tokenize(self):
        tokens = []
        while self.current_char:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in LETTERS and self.current_char != 'x':
                tokens.append(self.create_function())
            elif self.current_char in DIGITS:
                tokens.append(self.create_number())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*' or self.current_char == 'x':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tokens.append(Token(TT_FACT, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + self.current_char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def create_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def create_function(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS:
            id_str += self.current_char
            self.advance()

        tok_type = TT_FUNC
        return Token(tok_type, id_str, pos_start, self.pos)


# NODES
class NumberNode:
    def __init__(self, token):
        self.token = token
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'


class BinOpNode:
    def __init__(self, lnode, op_token, rnode):
        self.lnode = lnode
        self.op_token = op_token
        self.rnode = rnode
        self.pos_start = self.lnode.pos_start
        self.pos_end = self.rnode.pos_end

    def __repr__(self):
        return f'({self.lnode}, {self.op_token}, {self.rnode})'


class UnaryOpNode:
    def __init__(self, op_token, node):
        self.op_token = op_token
        self.node = node
        self.pos_start = self.op_token.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'({self.op_token}, {self.node})'


class FuncNode:
    def __init__(self, func_name, value):
        self.func_name = func_name
        self.value = value
        self.pos_start = self.func_name.pos_start
        self.pos_end = self.value.pos_end

    def __repr__(self):
        return f'({self.func_name}, {self.value})'


# PARSE RESULT
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, result):
        if isinstance(result, ParseResult):
            if result.error: self.error = result.error
            return result.node

        return result

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


# PARSER
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_index = -1
        self.current_token = None
        self.advance()

    def advance(self):
        self.tok_index += 1
        if self.tok_index in range(len(self.tokens)):
            self.current_token = self.tokens[self.tok_index]
        return self.current_token

    def parse(self):
        result = self.expr()
        if not result.error and self.current_token.type != TT_EOF:
            return result.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end,
                                                     'Expected +, -, *, or /'))
        return result

    def atom(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            result.register(self.advance())
            return result.success(NumberNode(token))

        elif token.type == TT_FUNC:
            func = result.register(self.func())
            if result.error: return result
            result.register(self.advance())
            return result.success(func)

        elif token.type == TT_LPAREN:
            result.register(self.advance())
            expr = result.register(self.expr())
            if result.error: return result
            if self.current_token.type == TT_RPAREN:
                result.register(self.advance())
                return result.success(expr)
            else:
                return result.failure(InvalidSyntaxError(token.pos_start, token.pos_end, 'Expected )'))

        return result.failure(InvalidSyntaxError(token.pos_start, token.pos_end, "Expected Int, Float, '-' or '('"))

    def func(self):
        result = ParseResult()
        func_name = self.current_token
        result.register(self.advance())
        if self.current_token.type == TT_LPAREN:
            result.register(self.advance())
            value = result.register(self.expr())
            if self.current_token.type != TT_RPAREN:
                return result.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, 'Expected )'))
            else: return result.success(FuncNode(func_name, value))

        value = result.register(self.expr())
        return result.success(FuncNode(func_name, value))

    def power(self):
        return self.bin_op(self.atom, (TT_POW, ), self.factor)

    def factor(self):
        result = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS, TT_FACT):
            result.register(self.advance())
            factor = result.register(self.factor())
            if result.error: return result
            return result.success(UnaryOpNode(token, factor))

        return self.power()

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func_a, ops, func_b=None):
        result = ParseResult()
        left = result.register(func_a())
        if result.error: return result

        while self.current_token.type in ops:
            op_token = self.current_token
            result.register(self.advance())
            right = result.register(func_a())
            if result.error: return result
            left = BinOpNode(left, op_token, right)

        return result.success(left)


# VALUES
class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def __repr__(self):
        return str(self.value)

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def add(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def sub(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def mul(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def div(self, other):
        if isinstance(other, Number):
            if other.value == 0: return None, RTError(other.pos_start, other.pos_end, 'Division by zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None

    def pow(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def fact(self):
        return Number(math.factorial(self.value)).set_context(self.context), None

    def sin(self):
        return Number((math.sin(self.value * math.pi / 180))).set_context(self.context), None

    def cos(self):
        return Number((math.cos(self.value * math.pi / 180))).set_context(self.context), None

    def tan(self):
        return Number((math.tan(self.value * math.pi / 180))).set_context(self.context), None

    def csc(self):
        return Number(1 / (math.sin(self.value * math.pi / 180))).set_context(self.context), None

    def sec(self):
        return Number(1 / (math.cos(self.value * math.pi / 180))).set_context(self.context), None

    def cot(self):
        return Number(1 / (math.tan(self.value * math.pi / 180))).set_context(self.context), None

    def asin(self):
        try:
            return Number(math.asin(self.value) * 180 / math.pi).set_context(self.context), None
        except ValueError: return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arcsin', self.context)

    def acos(self):
        try:
            return Number(math.acos(self.value) * 180 / math.pi).set_context(self.context), None
        except ValueError:
            return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arccos', self.context)

    def atan(self):
        try:
            return Number(math.atan(self.value) * 180 / math.pi).set_context(self.context), None
        except ValueError:
            return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arctan', self.context)

    def acsc(self):
        try:
            return Number(1 / (math.asin(self.value)) * 180 / math.pi).set_context(self.context), None
        except ValueError:
            return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arccsc', self.context)

    def asec(self):
        try:
            return Number(1 / (math.acos(self.value)) * 180 / math.pi).set_context(self.context), None
        except ValueError:
            return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arcsec', self.context)

    def acot(self):
        try:
            return Number(1 / (math.atan(self.value)) * 180 / math.pi).set_context(self.context), None
        except ValueError:
            return None, RTError(self.pos_start, self.pos_end, 'Invalid input for arccot', self.context)

    def root(self):
        if self.value < 0: return None, RTError(self.pos_start, self.pos_end, 'Negative radicand for a square root', self.context)
        return Number(math.sqrt(self.value)).set_context(self.context), None

    def ln(self):
        if self.value <= 0: return None, RTError(self.pos_start, self.pos_end, 'Negative or zero value for a natural log', self.context)
        return Number(math.log(self.value)).set_context(self.context), None

    def log(self):
        if self.value <= 0: return None, RTError(self.pos_start, self.pos_end, 'Negative or zero value for a logarithm', self.context)
        return Number(math.log10(self.value)).set_context(self.context), None


# CONTEXT
class Context:
    def __init__(self, name, parent=None, parent_pos=None):
        self.name = name
        self.parent = parent
        self.parent_pos = parent_pos


# INTERPRETER
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        # visit_BinOpNode
        # visit_NumberNode
        # visit_UnaryOpNode
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node, context):
        rt = RTResult()
        left = rt.register(self.visit(node.lnode, context))
        right = rt.register(self.visit(node.rnode, context))
        if rt.error: return rt
        result = None
        error = None

        if node.op_token.type == TT_PLUS:
            result, error = left.add(right)
        elif node.op_token.type == TT_MINUS:
            result, error = left.sub(right)
        elif node.op_token.type == TT_MUL:
            result, error = left.mul(right)
        elif node.op_token.type == TT_DIV:
            result, error = left.div(right)
        elif node.op_token.type == TT_POW:
            result, error = left.pow(right)

        if error:
            return rt.failure(error)
        else:
            return rt.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        rt = RTResult()
        number = rt.register(self.visit(node.node, context))
        error = None

        if node.op_token.type == TT_MINUS:
            number, error = number.mul(Number(-1))
        elif node.op_token.type == TT_FACT:
            number, error = number.fact()

        if error:
            return rt.failure(error)
        else:
            return rt.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_FuncNode(self, node, context):
        rt = RTResult()
        # name = rt.register(self.visit(node.func_name, context))
        name = node.func_name
        val = rt.register(self.visit(node.value, context))
        # val = node.value
        if rt.error: return rt
        result = None
        error = None
        if 'sin' in str(name):
            result, error = val.sin()
        if 'cos' in str(name):
            result, error = val.cos()
        if 'tan' in str(name):
            result, error = val.tan()
        if 'csc' in str(name):
            result, error = val.csc()
        if 'sec' in str(name):
            result, error = val.sec()
        if 'cot' in str(name):
            result, error = val.cot()
        if 'asin' in str(name) or 'arcsin' in str(name):
            result, error = val.asin()
        if 'acos' in str(name) or 'arccos' in str(name):
            result, error = val.acos()
        if 'atan' in str(name) or 'arctan' in str(name):
            result, error = val.atan()
        if 'acsc' in str(name) or 'arccsc' in str(name):
            result, error = val.acsc()
        if 'asec' in str(name) or 'arcsec' in str(name):
            result, error = val.asec()
        if 'acot' in str(name) or 'arccot' in str(name):
            result, error = val.acot()
        if 'root' in str(name):
            result, error = val.root()
        if 'ln' in str(name):
            result, error = val.ln()
        if 'log' in str(name):
            result, error = val.log()
        if error:
            return rt.failure(error)
        else:
            return rt.success(result.set_pos(node.func_name.pos_start, node.value.pos_end))


# RUN
def run(file_name, line):
    # Create tokens
    lex = Lexer(file_name, line)
    tokens, error = lex.tokenize()
    if error: return None, error

    # Create abstract syntax tree
    parser = Parser(tokens)
    tree = parser.parse()
    if tree.error: return None, tree.error

    # Run program
    interpreter = Interpreter()
    context = Context('<Calculator>')
    result = interpreter.visit(tree.node, context)

    return result.value, result.error


while True:
    line = input('> ')
    if line == 'quit':
        exit()
    result, error = run('Calculator:', line)

    if error: print(error.as_string())
    else: print(result)
