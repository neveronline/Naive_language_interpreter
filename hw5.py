import sys

tokens = (

    'NAME', 'NUMBER', 'FLOAT', 'STRING',

    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'EQUALS','SMALLER','GREATER','AND','OR','NOT','MOD','EXPONENT','FDIV','IN',

    'LPAREN', 'RPAREN','LBRACKET','RBRACKET','QUOTE','COMMA','SMALLEREQUAL','GREATEREQUAL','NOTEQUAL','DOUBLEEQUAL',
    'PRINT','SEMI','LCBRACE','RCBRACE','IF','WHILE','ELSE'

)





# Tokens

t_PLUS = r'\+'

t_MINUS = r'-'

t_TIMES = r'\*'

t_DIVIDE = r'/'

t_EQUALS = r'='

t_LPAREN = r'\('

t_RPAREN = r'\)'



t_LBRACKET =r'\['

t_RBRACKET = r'\]'

t_QUOTE =r'\"'

t_SMALLER =r'\<'

t_GREATER =r'>'

t_COMMA =r'\,'

t_MOD = r'\%'

t_EXPONENT = r'\*\*'

t_FDIV = r'//'

t_SMALLEREQUAL = r'\<='

t_GREATEREQUAL = r'\>='

t_NOTEQUAL = r'\<\>'

t_DOUBLEEQUAL = r'=='

t_SEMI = r';'

t_LCBRACE = r'\{'

t_RCBRACE =r'\}'



def t_ELSE(t):
    r'else'
    t.value = 'else'
    return t

def t_WHILE(t):
    r'while'
    t.value = 'while'
    return t

def t_IF(t):
    r'if'
    t.value = 'if'
    return t

def t_NOT(t):
    r'not'
    return t

def t_AND(t):
    r'and'
    return t

def t_OR(t):
    r'or'
    return t
def t_IN(t):
    r'in'
    return t

def t_PRINT(t):
    r'print'
    t.value = 'print'
    return t

def t_STRING(t):
    r'\"([^\\\n]|(\\.))*?\"'
    t.value = str(t.value)
    return t


t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_NUMBER(t):
    r'\d+'

    try:

        t.value = NumberNode(t.value)

    except ValueError:

        print("Integer value too large %d", t.value)

        t.value = 0


    return t


# Ignored characters

t_ignore = " \t"


def t_newline(t):
    r'\n+'

    t.lexer.lineno += t.value.count("\n")



def t_error(t):
    print("Illegal character '%s'" % t.value[0])

    t.lexer.skip(1)

# Build the lexer

import ply.lex as lex

lex.lex()

# Parsing rules

precedence = (
    ('left','OR'),
    ('left','AND'),
    ('left','NOT'),
    ('left', 'IN'),
    ('left', 'GREATER', 'SMALLER', 'GREATEREQUAL', 'SMALLEREQUAL', 'DOUBLEEQUAL','NOTEQUAL'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'FDIV'),
    ('left', 'EXPONENT'),
    ('left', 'MOD'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right','UMINUS'),
    ('left','LBRACKET','RBRACKET','COMMA'),
    ('left', 'LPAREN', 'RPAREN')
)

# dictionary of names

names = {}


def p_statement_block(p):
    ''' statement : LCBRACE blocks RCBRACE '''
    p[0] = BlockNode(p[2])


def p_blocks(p):
    """blocks : block blocksl"""

    p[0] = [p[1]]+p[2]


def p_blocksl(p):
    """blocksl :  block blocksl
                """

    p[0] = [p[1]] + p[2]



def p_control_while(p):
    """block : WHILE LPAREN expression RPAREN statement """
    p[0] = WHILENode(p[3], p[5])

def p_control_if(p):
    """block : IF LPAREN expression RPAREN statement
               | IF LPAREN expression RPAREN statement ELSE statement"""
    if len(p) == 6:  # We have no else block
        p[0] = IFNode(p[3], p[5],None)

    elif len(p) == 8:  # We have an else block
        p[0] = IFNode(p[3], p[5], elseblock=p[7])


def p_blockll(p):
    """blocksl :
                | SEMI

                """
    p[0] =[]


def p_blocks_empty(p):
    '''blocks :
    '''

def p_blk_expr(p):
    'block : expression'

    p[0] = p[1]



def p_statement_print(p):
    ''' expression : PRINT LPAREN expression RPAREN SEMI'''
    p[0] = PrintNode(p[3])


def p_expression_assign(p):
    'expression : NAME EQUALS expression SEMI'

    p[0] = AssignNode(p[1], p[3])




def p_expression_binop(t):
    '''expression : expression PLUS expression

                  | expression MINUS expression

                  | expression TIMES expression

                  | expression DIVIDE expression

                  | expression MOD expression

                  | expression EXPONENT expression

                  | expression FDIV expression
                  '''
    t[0] = opNode(t[1],t[2],t[3])




def p_expression_uminus(t):
    'expression : MINUS expression %prec UMINUS'

    t[0] = -t[2]


def p_expression_list_assign(p):
    """expression : expression retrive EQUALS NAME SEMI"""
    p[0] = AssignListNode(p[1],p[2],VarNode(p[4]))

def p_list_extract(t):
    '''
    expression : expression retrive
    '''

    t[0] = IndexedVar(t[1],t[2])


def p_list_retrive(t):
    '''
    retrive : LBRACKET NAME RBRACKET
    '''

    t[0] = VarNode(t[2])

def p_retrive_from_string(t):
    '''
    retrive : LBRACKET NUMBER RBRACKET


    '''
    t[0] = t[2]



def p_boolean(t):
    '''
    expression : expression SMALLER expression
                | expression GREATER expression
                | expression SMALLEREQUAL expression
                | expression GREATEREQUAL expression
                | expression NOTEQUAL expression
                | expression DOUBLEEQUAL expression

    '''
    t[0] = opNode(t[1], t[2], t[3])

def p_logic_operation(t):
    '''
    expression : expression AND expression
               | expression OR expression
               | expression IN expression


    '''

    t[0] = opNode(t[1],t[2],t[3])

def p_not (t):
    '''
    expression : NOT expression
    '''
    t[0] = opNode(t[1],t[2],None)

def p_expression_number(t):
    '''expression : NUMBER
                    | FLOAT


    '''

    t[0] = t[1]

def p_expression_string(t):
    '''
    expression : STRING

    '''
    t[0] = StringNode(str(t[1]))

def p_expression_name(t):
    '''expression : NAME

    '''
    t[0] = VarNode(t[1])


def p_expression_list(p):
    """expression : list
    """
    p[0] = ListNode(p[1])


def p_list(p):
    '''
    list : LBRACKET list_tail RBRACKET

    '''
    if len(p)==3:
        p[0]=[]
    else:
        p[0] = p[2]

def p_list_tail(t):
    '''
    list_tail : expression
            | expression COMMA list_tail
    '''

    if len(t) == 4:
        t[0] =  [t[1]]+ t[3]
    else:
        t[0] = [t[1]]



def p_element_in_list(t):
    '''
    expression : list LBRACKET expression RBRACKET
    '''

    t[0] = list(t[1])[t[3].execute()]




def p_error(t):
    print("Syntax error at '%s'" % t.value)

# def p_blocks_empty(p):
#     '''blocks :
#     '''



class Node:
    def __init__(self):
        print("Node")

    def evaluate(self):
        print("Evaluate")
        return 0

    def execute(self):
        print("Execute")

class ListNode:
    def __init__(self,list):
        self.list = list

    def evaluate(self):

        return [x.evaluate() for x in self.list]

    def execute(self):

        return [x.evaluate() for x in self.list]

class VarNode(Node):
    def __init__(self,v):
        self.value = v


    def evaluate(self):
        return names[self.value]
    def execute(self):
        return self.value

class AssignNode(Node):
    def __init__(self,name,value):
        self.name = name
        self.value = value

    def evaluate(self):
        return 0
    def execute(self):
        names[self.name] = self.value.evaluate()

class IndexedVar(Node):
    def __init__(self,exp,index):
        self.exp = exp
        self.index = index

    def evaluate(self):


        return self.exp.evaluate()[self.index.evaluate()]



    def execute(self):

        return self.evaluate()


class AssignListNode(Node):
    def __init__(self,exp,index,newValue):
        self.exp = exp
        self.index = index
        self.value = newValue

    def evaluate(self):
        # print("assign list element a new value")
        self.exp.evaluate()[self.index.evaluate()] = self.value.evaluate()

    def execute(self):
        # print(self.exp.evaluate())
        return self.evaluate()


class opNode(Node):
    def __init__(self,left,op,right):
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self):
        if self.right == None:
            if self.left == 'not':
                return not self.op

        if self.op == '*':
            return self.left.evaluate() * self.right.evaluate()
        elif self.op == '/':
            return self.left.evaluate() / self.right.evaluate()
        elif self.op == '+':
            return self.left.evaluate() + self.right.evaluate()
        elif self.op == '-':
            return self.left.evaluate() - self.right.evaluate()
        elif self.op == '>':
            return self.left.evaluate() > self.right.evaluate()
        elif self.op == '<':
            return self.left.evaluate() < self.right.evaluate()
        elif self.op == '>=':
            return self.left.evaluate() >= self.right.evaluate()
        elif self.op == '<=':
            return self.left.evaluate() <= self.right.evaluate()
        elif self.op == '%':
            return self.left.evaluate() % self.right.evaluate()
        elif self.op == '**':
            return self.left.evaluate() ** self.right.evaluate()
        elif self.op == '//':
            return self.left.evaluate() // self.right.evaluate()
        elif self.op == '<>':
            return self.left.evaluate() != self.right.evaluate()
        elif self.op == '==':
            return self.left.evaluate() == self.right.evaluate()
        elif self.op == 'and':
            return self.left.evaluate() and self.right.evaluate()
        elif self.op == 'or':
            return self.left.evaluate() or self.right.evaluate()
        elif self.op == 'in':
            return self.left.evaluate() in self.right.evaluate()

    def execute(self):
        self.evaluate()

class NumberNode(Node):
    def __init__(self, v):
        self.value = int(v)


    def evaluate(self):

        return self.value

    def execute(self):
        self.evaluate()



class StringNode(Node):
    def __init__(self, v):
        self.value = str(v)
        self.value = self.value[1:-1]  # to eliminate the left and right double quotes


    def evaluate(self):

        return self.value

    def execute(self):
        return self.value
        print("Execute StringNode")


class PrintNode(Node):
    def __init__(self, v):
        self.value = v


    def evaluate(self):

        return 0

    def execute(self):

        print(self.value.evaluate())


class IFNode(Node):
    def __init__(self, condition, block, elseblock=None):

        self.condition = condition
        self.block = block
        self.elseblock = elseblock
    def evaluate(self):
        if bool(self.condition.evaluate()):

            self.block.execute()
        elif self.elseblock is not None:
            self.elseblock.execute()

    def execute(self):
        self.evaluate()

class BlockNode(Node):
    def __init__(self, sl):
        self.statementNodes = sl


    def evaluate(self):

        return 0

    def execute(self):

        for statement in self.statementNodes:
            statement.execute()

class WHILENode(Node):
    def __init__(self, condition, block):

        self.condition = condition
        self.block = block
    def execute(self):
        while bool(self.condition.evaluate()):
            self.block.execute()




import ply.yacc as yacc

yacc.yacc()


# while 1:
#
#     try:
#
#         s = input("calc > ")   # Use raw_input on Python 2
#
#     except EOFError:
#
#         break
#
#     yacc.parse(s).execute()

filename = sys.argv[1]
f = open(filename,'r')
code = ""
for line in f:
    code+= line
yacc.parse(code).execute()