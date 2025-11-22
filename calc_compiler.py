import re
import sys

# ============================================================================
# LEXER - Tokenizes input source code
# ============================================================================

TOKEN_SPEC = [
    ('NUMBER',   r'\d+'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('EQ',       r'=='),
    ('NEQ',      r'!='),
    ('LE',       r'<='),
    ('GE',       r'>='),
    ('LT',       r'<'),
    ('GT',       r'>'),
    ('ASSIGN',   r'='),
    ('PLUS',     r'\+'),
    ('MINUS',    r'-'),
    ('MUL',      r'\*'),
    ('DIV',      r'/'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('SEMI',     r';'),
    ('COMMA',    r','),
    ('SKIP',     r'[ \t\r\n]+'),
    ('COMMENT',  r'//.*'),
]

KEYWORDS = {
    'print': 'PRINT',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'generate': 'GENERATE',
    'sequence': 'SEQUENCE',
    'from': 'FROM',
    'to': 'TO',
    'step': 'STEP'
}

class Token:
    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"

def lex(code):
    """Tokenize the input code."""
    token_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)
    line_num = 1
    line_start = 0
    tokens = []
    
    for match in re.finditer(token_pattern, code):
        kind = match.lastgroup
        value = match.group()
        column = match.start() - line_start
        
        if kind == 'NUMBER':
            tokens.append(Token('NUMBER', int(value), line_num, column))
        elif kind == 'ID':
            token_type = KEYWORDS.get(value, 'ID')
            tokens.append(Token(token_type, value, line_num, column))
        elif kind == 'SKIP':
            line_num += value.count('\n')
            if '\n' in value:
                line_start = match.end()
        elif kind == 'COMMENT':
            pass  # Ignore comments
        else:
            tokens.append(Token(kind, value, line_num, column))
    
    tokens.append(Token('EOF', None, line_num, len(code)))
    return tokens

# ============================================================================
# AST NODES - Abstract Syntax Tree representation
# ============================================================================

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class Assignment(ASTNode):
    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression

class PrintStmt(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class IfStmt(ASTNode):
    def __init__(self, condition, then_block, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class WhileStmt(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class SequenceStmt(ASTNode):
    def __init__(self, name, values):
        self.name = name
        self.values = values

class BinaryOp(ASTNode):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

class UnaryOp(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

class Number(ASTNode):
    def __init__(self, value):
        self.value = value

class Variable(ASTNode):
    def __init__(self, name):
        self.name = name

# ============================================================================
# PARSER - Builds AST from tokens
# ============================================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self):
        return self.tokens[self.pos]
    
    def consume(self, expected_type=None):
        token = self.current_token()
        if expected_type and token.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token.type} at line {token.line}")
        self.pos += 1
        return token
    
    def parse(self):
        """Parse the entire program."""
        statements = []
        while self.current_token().type != 'EOF':
            statements.append(self.parse_statement())
        return Program(statements)
    
    def parse_statement(self):
        """Parse a single statement."""
        token = self.current_token()
        
        if token.type == 'PRINT':
            return self.parse_print()
        elif token.type == 'IF':
            return self.parse_if()
        elif token.type == 'WHILE':
            return self.parse_while()
        elif token.type == 'GENERATE':
            return self.parse_generate()
        elif token.type == 'SEQUENCE':
            return self.parse_sequence()
        elif token.type == 'LBRACE':
            return self.parse_block()
        elif token.type == 'ID':
            return self.parse_assignment()
        else:
            raise SyntaxError(f"Unexpected token {token.type} at line {token.line}")
    
    def parse_assignment(self):
        """Parse: ID = expression;"""
        var_name = self.consume('ID').value
        self.consume('ASSIGN')
        expr = self.parse_expression()
        self.consume('SEMI')
        return Assignment(var_name, expr)
    
    def parse_print(self):
        """Parse: print(expression);"""
        self.consume('PRINT')
        self.consume('LPAREN')
        expr = self.parse_expression()
        self.consume('RPAREN')
        self.consume('SEMI')
        return PrintStmt(expr)
    
    def parse_if(self):
        """Parse: if (condition) statement [else statement]"""
        self.consume('IF')
        self.consume('LPAREN')
        condition = self.parse_expression()
        self.consume('RPAREN')
        then_stmt = self.parse_statement()
        else_stmt = None
        if self.current_token().type == 'ELSE':
            self.consume('ELSE')
            else_stmt = self.parse_statement()
        return IfStmt(condition, then_stmt, else_stmt)
    
    def parse_while(self):
        """Parse: while (condition) statement"""
        self.consume('WHILE')
        self.consume('LPAREN')
        condition = self.parse_expression()
        self.consume('RPAREN')
        body = self.parse_statement()
        return WhileStmt(condition, body)
    
    def parse_generate(self):
        """Parse: generate var from start to end step increment;"""
        self.consume('GENERATE')
        var_name = self.consume('ID').value
        self.consume('FROM')
        start = self.parse_expression()
        self.consume('TO')
        end = self.parse_expression()
        self.consume('STEP')
        step = self.parse_expression()
        self.consume('SEMI')
        
        # Transform into: var = start; while(var <= end) { print(var); var = var + step; }
        init = Assignment(var_name, start)
        condition = BinaryOp('LE', Variable(var_name), end)
        print_stmt = PrintStmt(Variable(var_name))
        increment = Assignment(var_name, BinaryOp('PLUS', Variable(var_name), step))
        loop_body = Block([print_stmt, increment])
        loop = WhileStmt(condition, loop_body)
        
        return Block([init, loop])
    
    def parse_sequence(self):
        """Parse: sequence name = expr1, expr2, ...;"""
        self.consume('SEQUENCE')
        name = self.consume('ID').value
        self.consume('ASSIGN')
        values = []
        values.append(self.parse_expression())
        while self.current_token().type == 'COMMA':
            self.consume('COMMA')
            values.append(self.parse_expression())
        self.consume('SEMI')
        return SequenceStmt(name, values)
    
    def parse_block(self):
        """Parse: { statement* }"""
        self.consume('LBRACE')
        statements = []
        while self.current_token().type != 'RBRACE':
            statements.append(self.parse_statement())
        self.consume('RBRACE')
        return Block(statements)
    
    def parse_expression(self):
        """Parse expressions with operator precedence."""
        return self.parse_equality()
    
    def parse_equality(self):
        """Parse: == !="""
        left = self.parse_comparison()
        while self.current_token().type in ('EQ', 'NEQ'):
            op = self.consume().type
            right = self.parse_comparison()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_comparison(self):
        """Parse: < > <= >="""
        left = self.parse_additive()
        while self.current_token().type in ('LT', 'GT', 'LE', 'GE'):
            op = self.consume().type
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_additive(self):
        """Parse: + -"""
        left = self.parse_multiplicative()
        while self.current_token().type in ('PLUS', 'MINUS'):
            op = self.consume().type
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_multiplicative(self):
        """Parse: * /"""
        left = self.parse_unary()
        while self.current_token().type in ('MUL', 'DIV'):
            op = self.consume().type
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_unary(self):
        """Parse: -expr"""
        if self.current_token().type == 'MINUS':
            self.consume('MINUS')
            return UnaryOp('MINUS', self.parse_unary())
        return self.parse_primary()
    
    def parse_primary(self):
        """Parse: number | variable | (expression)"""
        token = self.current_token()
        
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return Number(token.value)
        elif token.type == 'ID':
            self.consume('ID')
            return Variable(token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        else:
            raise SyntaxError(f"Unexpected token {token.type} at line {token.line}")

# ============================================================================
# SEMANTIC ANALYZER - Type checking and symbol table
# ============================================================================

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}
        self.errors = []
    
    def analyze(self, node):
        """Analyze the AST."""
        method_name = f'analyze_{node.__class__.__name__}'
        method = getattr(self, method_name, self.generic_analyze)
        return method(node)
    
    def generic_analyze(self, node):
        raise Exception(f"No analyze method for {node.__class__.__name__}")
    
    def analyze_Program(self, node):
        for stmt in node.statements:
            self.analyze(stmt)
    
    def analyze_Block(self, node):
        for stmt in node.statements:
            self.analyze(stmt)
    
    def analyze_Assignment(self, node):
        self.analyze(node.expression)
        self.symbol_table[node.variable] = 'int'
    
    def analyze_PrintStmt(self, node):
        self.analyze(node.expression)
    
    def analyze_IfStmt(self, node):
        self.analyze(node.condition)
        self.analyze(node.then_block)
        if node.else_block:
            self.analyze(node.else_block)
    
    def analyze_WhileStmt(self, node):
        self.analyze(node.condition)
        self.analyze(node.body)
    
    def analyze_SequenceStmt(self, node):
        for expr in node.values:
            self.analyze(expr)
        self.symbol_table[node.name] = 'sequence'
    
    def analyze_BinaryOp(self, node):
        self.analyze(node.left)
        self.analyze(node.right)
    
    def analyze_UnaryOp(self, node):
        self.analyze(node.operand)
    
    def analyze_Number(self, node):
        pass
    
    def analyze_Variable(self, node):
        if node.name not in self.symbol_table:
            self.errors.append(f"Variable '{node.name}' used before assignment")

# ============================================================================
# CODE GENERATOR - Generates Three-Address Code (TAC)
# ============================================================================

class CodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self):
        """Generate a new temporary variable."""
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self):
        """Generate a new label."""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def emit(self, op, arg1=None, arg2=None, result=None):
        """Emit a three-address code instruction."""
        self.code.append((op, arg1, arg2, result))
    
    def generate(self, node):
        """Generate code for the AST."""
        method_name = f'generate_{node.__class__.__name__}'
        method = getattr(self, method_name)
        return method(node)
    
    def generate_Program(self, node):
        for stmt in node.statements:
            self.generate(stmt)
        return self.code
    
    def generate_Block(self, node):
        for stmt in node.statements:
            self.generate(stmt)
    
    def generate_Assignment(self, node):
        expr_result = self.generate(node.expression)
        self.emit('=', expr_result, None, node.variable)
    
    def generate_PrintStmt(self, node):
        expr_result = self.generate(node.expression)
        self.emit('print', expr_result, None, None)
    
    def generate_IfStmt(self, node):
        cond_result = self.generate(node.condition)
        else_label = self.new_label()
        end_label = self.new_label()
        
        self.emit('ifFalse', cond_result, None, else_label)
        self.generate(node.then_block)
        self.emit('goto', None, None, end_label)
        self.emit('label', None, None, else_label)
        if node.else_block:
            self.generate(node.else_block)
        self.emit('label', None, None, end_label)
    
    def generate_WhileStmt(self, node):
        start_label = self.new_label()
        end_label = self.new_label()
        
        self.emit('label', None, None, start_label)
        cond_result = self.generate(node.condition)
        self.emit('ifFalse', cond_result, None, end_label)
        self.generate(node.body)
        self.emit('goto', None, None, start_label)
        self.emit('label', None, None, end_label)
    
    def generate_SequenceStmt(self, node):
        for expr in node.values:
            result = self.generate(expr)
            self.emit('print', result, None, None)
    
    def generate_BinaryOp(self, node):
        left_result = self.generate(node.left)
        right_result = self.generate(node.right)
        result = self.new_temp()
        
        op_map = {
            'PLUS': '+', 'MINUS': '-', 'MUL': '*', 'DIV': '/',
            'EQ': '==', 'NEQ': '!=', 'LT': '<', 'GT': '>',
            'LE': '<=', 'GE': '>='
        }
        op = op_map.get(node.operator, node.operator)
        self.emit(op, left_result, right_result, result)
        return result
    
    def generate_UnaryOp(self, node):
        operand_result = self.generate(node.operand)
        result = self.new_temp()
        self.emit('neg', operand_result, None, result)
        return result
    
    def generate_Number(self, node):
        result = self.new_temp()
        self.emit('const', node.value, None, result)
        return result
    
    def generate_Variable(self, node):
        return node.name

# ============================================================================
# OPTIMIZER - Optimizes Three-Address Code
# ============================================================================

class Optimizer:
    def __init__(self, code):
        self.code = code
    
    def constant_folding(self):
        """Perform constant folding optimization."""
        constants = {}
        optimized = []
        
        for op, arg1, arg2, result in self.code:
            if op == 'const':
                constants[result] = arg1
                optimized.append((op, arg1, arg2, result))
            elif op in ('+', '-', '*', '/'):
                val1 = constants.get(arg1)
                val2 = constants.get(arg2)
                if val1 is not None and val2 is not None:
                    if op == '+':
                        value = val1 + val2
                    elif op == '-':
                        value = val1 - val2
                    elif op == '*':
                        value = val1 * val2
                    elif op == '/' and val2 != 0:
                        value = val1 // val2
                    else:
                        value = 0
                    constants[result] = value
                    optimized.append(('const', value, None, result))
                else:
                    optimized.append((op, arg1, arg2, result))
            else:
                optimized.append((op, arg1, arg2, result))
        
        self.code = optimized
        return self.code
    
    def dead_code_elimination(self):
        """Remove unused temporary variables."""
        used = set()
        
        # Find all used variables
        for op, arg1, arg2, result in self.code:
            if op in ('print', 'ifFalse', '='):
                if arg1:
                    used.add(arg1)
            if op == '=':
                if arg2:
                    used.add(arg2)
        
        # Backward pass to find dependencies
        changed = True
        while changed:
            changed = False
            for op, arg1, arg2, result in reversed(self.code):
                if result and result in used:
                    if arg1 and arg1 not in used:
                        used.add(arg1)
                        changed = True
                    if arg2 and arg2 not in used:
                        used.add(arg2)
                        changed = True
        
        # Remove dead code
        optimized = []
        for instr in self.code:
            op, arg1, arg2, result = instr
            if result and result.startswith('t') and result not in used:
                continue
            optimized.append(instr)
        
        self.code = optimized
        return self.code

# ============================================================================
# INTERPRETER - Executes Three-Address Code
# ============================================================================

class Interpreter:
    def __init__(self):
        self.variables = {}
        self.code = []
        self.pc = 0
        self.labels = {}
    
    def load(self, code):
        """Load TAC code for execution."""
        self.code = code
        self.pc = 0
        self.labels = {}
        
        # Find all labels
        for i, (op, arg1, arg2, result) in enumerate(code):
            if op == 'label':
                self.labels[result] = i
    
    def get_value(self, operand):
        """Get the value of an operand (variable or constant)."""
        if operand is None:
            return None
        if isinstance(operand, int):
            return operand
        return self.variables.get(operand, 0)
    
    def execute(self):
        """Execute the loaded TAC code."""
        while self.pc < len(self.code):
            op, arg1, arg2, result = self.code[self.pc]
            
            if op == 'const':
                self.variables[result] = arg1
            elif op == '=':
                self.variables[result] = self.get_value(arg1)
            elif op == '+':
                self.variables[result] = self.get_value(arg1) + self.get_value(arg2)
            elif op == '-':
                self.variables[result] = self.get_value(arg1) - self.get_value(arg2)
            elif op == '*':
                self.variables[result] = self.get_value(arg1) * self.get_value(arg2)
            elif op == '/':
                val2 = self.get_value(arg2)
                self.variables[result] = self.get_value(arg1) // val2 if val2 != 0 else 0
            elif op == 'neg':
                self.variables[result] = -self.get_value(arg1)
            elif op == '==':
                self.variables[result] = 1 if self.get_value(arg1) == self.get_value(arg2) else 0
            elif op == '!=':
                self.variables[result] = 1 if self.get_value(arg1) != self.get_value(arg2) else 0
            elif op == '<':
                self.variables[result] = 1 if self.get_value(arg1) < self.get_value(arg2) else 0
            elif op == '>':
                self.variables[result] = 1 if self.get_value(arg1) > self.get_value(arg2) else 0
            elif op == '<=':
                self.variables[result] = 1 if self.get_value(arg1) <= self.get_value(arg2) else 0
            elif op == '>=':
                self.variables[result] = 1 if self.get_value(arg1) >= self.get_value(arg2) else 0
            elif op == 'print':
                print(self.get_value(arg1))
            elif op == 'ifFalse':
                if self.get_value(arg1) == 0:
                    self.pc = self.labels[result]
                    continue
            elif op == 'goto':
                self.pc = self.labels[result]
                continue
            elif op == 'label':
                pass
            
            self.pc += 1

# ============================================================================
# MAIN COMPILER PIPELINE
# ============================================================================

def compile_and_run(source_code, show_phases=False, optimize=True):
    """Complete compilation pipeline."""
    try:
        # Phase 1: Lexical Analysis
        if show_phases:
            print("=" * 60)
            print("PHASE 1: LEXICAL ANALYSIS")
            print("=" * 60)
        tokens = lex(source_code)
        if show_phases:
            for token in tokens[:-1]:  # Exclude EOF
                print(f"  {token}")
            print()
        
        # Phase 2: Syntax Analysis (Parsing)
        if show_phases:
            print("=" * 60)
            print("PHASE 2: SYNTAX ANALYSIS (PARSING)")
            print("=" * 60)
            print("  Building Abstract Syntax Tree...")
        parser = Parser(tokens)
        ast = parser.parse()
        if show_phases:
            print("  AST built successfully")
            print()
        
        # Phase 3: Semantic Analysis
        if show_phases:
            print("=" * 60)
            print("PHASE 3: SEMANTIC ANALYSIS")
            print("=" * 60)
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        if analyzer.errors:
            for error in analyzer.errors:
                print(f"  Semantic Error: {error}")
            return None
        if show_phases:
            print("  Symbol Table:")
            for var, typ in analyzer.symbol_table.items():
                print(f"    {var}: {typ}")
            print()
        
        # Phase 4: Code Generation
        if show_phases:
            print("=" * 60)
            print("PHASE 4: CODE GENERATION (TAC)")
            print("=" * 60)
        generator = CodeGenerator()
        tac_code = generator.generate(ast)
        if show_phases:
            print("  Three-Address Code (before optimization):")
            for i, instr in enumerate(tac_code):
                print(f"    {i:3d}: {instr}")
            print()
        
        # Phase 5: Optimization
        if optimize:
            if show_phases:
                print("=" * 60)
                print("PHASE 5: OPTIMIZATION")
                print("=" * 60)
            optimizer = Optimizer(tac_code)
            tac_code = optimizer.constant_folding()
            tac_code = optimizer.dead_code_elimination()
            if show_phases:
                print("  Three-Address Code (after optimization):")
                for i, instr in enumerate(tac_code):
                    print(f"    {i:3d}: {instr}")
                print()
        
        # Phase 6: Execution
        if show_phases:
            print("=" * 60)
            print("PHASE 6: EXECUTION")
            print("=" * 60)
            print("  Output:")
        interpreter = Interpreter()
        interpreter.load(tac_code)
        interpreter.execute()
        
        return interpreter
        
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# REPL MODE
# ============================================================================

def repl_mode():
    """Interactive REPL mode for testing code snippets."""
    print("=" * 60)
    print("PatternLang Compiler - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  Type your code and press Enter")
    print("  ':phases' - Toggle showing compilation phases")
    print("  ':optimize' - Toggle optimization")
    print("  ':clear' - Clear interpreter state")
    print("  ':quit' or ':exit' - Exit REPL")
    print("\nExamples:")
    print("  x = 5; print(x);")
    print("  print(2 + 3 * 4);")
    print("  generate i from 1 to 5 step 1;")
    print("  if (x > 3) print(100); else print(200);")
    print()
    
    interpreter = Interpreter()
    show_phases = False
    optimize = True
    
    while True:
        try:
            line = input(">>> ").strip()
            
            if not line:
                continue
            
            # Handle commands
            if line.startswith(':'):
                cmd = line.lower()
                if cmd in (':quit', ':exit', ':q'):
                    print("Goodbye!")
                    break
                elif cmd == ':phases':
                    show_phases = not show_phases
                    print(f"Show phases: {show_phases}")
                    continue
                elif cmd == ':optimize':
                    optimize = not optimize
                    print(f"Optimization: {optimize}")
                    continue
                elif cmd == ':clear':
                    interpreter = Interpreter()
                    print("Interpreter state cleared")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Compile and execute
            result = compile_and_run(line, show_phases=show_phases, optimize=optimize)
            if result and not show_phases:
                print()
            
        except KeyboardInterrupt:
            print("\nUse :quit to exit")
        except EOFError:
            print("\nGoodbye!")
            break

# ============================================================================
# FILE MODE
# ============================================================================

def file_mode(filename, show_phases=False, optimize=True):
    """Compile and run a source file."""
    try:
        with open(filename, 'r') as f:
            source_code = f.read()
        
        print(f"Compiling: {filename}")
        print("=" * 60)
        compile_and_run(source_code, show_phases=show_phases, optimize=optimize)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PatternLang Compiler - A simple language compiler with full compilation phases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calc_compiler.py                    # Start interactive REPL
  python calc_compiler.py program.txt        # Compile and run a file
  python calc_compiler.py program.txt -p     # Show all compilation phases
  python calc_compiler.py program.txt -n     # Disable optimization
        """
    )
    
    parser.add_argument('source', nargs='?', help='Source file to compile')
    parser.add_argument('-p', '--phases', action='store_true', help='Show all compilation phases')
    parser.add_argument('-n', '--no-optimize', action='store_true', help='Disable optimization')
    
    args = parser.parse_args()
    
    if args.source:
        # File mode
        file_mode(args.source, show_phases=args.phases, optimize=not args.no_optimize)
    else:
        # REPL mode
        repl_mode()