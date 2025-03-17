from __future__ import annotations
from typing import List

import re
import sympy as sy

from .utils import timeout

SEP_TOKEN = '|'
SYMBOL_RE = re.compile(r"(x\d+)|(t)")
EXTRA_SYMBOL_RE = re.compile(r"(g\d+)") # intermediate symbols not appear in the final expression

BINARY_OPERATORS = {
    'add': sy.Add,
    'sub': (lambda x,y: x-y),
    'mul': sy.Mul,
    # 'div': (lambda x,y: x/y), # less frequent, replaced by 'inv'
    'pow': sy.Pow,
}

UNARY_OPERATORS = {
    'id'   : (lambda x: x),
    'inv'  : (lambda x: 1/x),
    'pow2' : (lambda x: x**2),
    'pow3' : (lambda x: x**3),
    'sqrt' : sy.sqrt,
    'exp'  : sy.exp,
    'log'  : sy.log,
    'sin'  : sy.sin,
    'cos'  : sy.cos,
    'tan'  : sy.tan,
    'cot'  : sy.cot,
    # 'asin' : sy.asin,
    # 'acos' : sy.acos,
    'atan' : sy.atan,
    # 'sinh' : sy.sinh,
    # 'cosh' : sy.cosh,
    # 'tanh' : sy.tanh,
    # 'asinh': sy.asinh,
    # 'acosh': sy.acosh,
    # 'atanh': sy.atanh,
    'abs'  : sy.Abs,
    # 'sign' : sy.sign,
    # 'step' : sy.Heaviside,
}

SYMPY_OPERATORS = {
    sy.Add  : 'add',
    sy.Mul  : 'mul',
    sy.Pow  : 'pow',
    sy.exp  : 'exp',
    sy.log  : 'log',
    sy.sin  : 'sin',
    sy.cos  : 'cos',
    sy.tan  : 'tan',
    sy.cot  : 'cot',
    # sy.asin : 'asin',
    # sy.acos : 'acos',
    sy.atan : 'atan',
    # sy.sinh : 'sinh',
    # sy.cosh : 'cosh',
    # sy.tanh : 'tanh',
    # sy.asinh: 'asinh',
    # sy.acosh: 'acosh',
    # sy.atanh: 'atanh',
    sy.Abs  : 'abs',
    # sy.sign : 'sign',
    # sy.Heaviside : 'step',
}

class SymbolicExpression():
    """
    Convertion between a sequence of prefix strings and a list of sympy expressions.
    """
    unary_operators  = UNARY_OPERATORS
    binary_operators = BINARY_OPERATORS
    operators = {**UNARY_OPERATORS, **BINARY_OPERATORS}
    sympy_operators = SYMPY_OPERATORS

    def __init__(self): pass

    @staticmethod
    def check_sequence(sequence: List[str]):
        assert all([isinstance(seq, str) for seq in sequence]), "expect a list of strings"
        
    def _sequence_to_sympy(self, sequence: List[str], symbols: dict):
        op = sequence.pop(0)
        if op in self.unary_operators:
            expr,sequence = self._sequence_to_sympy(sequence, symbols)
            return self.unary_operators[op](expr), sequence
        if op in self.binary_operators:
            expr1,sequence = self._sequence_to_sympy(sequence, symbols)
            expr2,sequence = self._sequence_to_sympy(sequence, symbols)
            return self.binary_operators[op](expr1, expr2), sequence
        if op in symbols:
            return symbols[op], sequence
        try:
            op = float(op)
        except:
            raise ValueError(f'{op} is not defined!')
        return op, sequence

    def sequence_to_sympy(self, sequence: List[str], get_np_func: bool=False, d: int=None):
        "Returns a list of sympy expressions from a list of prefix strings."
        SymbolicExpression.check_sequence(sequence)
        d = sum([seq == SEP_TOKEN for seq in sequence]) + 1 if not d else d # dimensionality

        symbols = list(set([seq for seq in sequence if (re.fullmatch(SYMBOL_RE, seq) or re.fullmatch(EXTRA_SYMBOL_RE, seq))]))
        symbols_dict = dict(zip(symbols, sy.symbols(symbols, real=True)))
        x_symbols = sy.symbols([f'x{i}' for i in range(1,d+1)], real=True) # variables (x1, ..., xd)
        
        exprs = []
        expr = []
        for seq in sequence+[SEP_TOKEN]:
            if seq == SEP_TOKEN:
                exprs.append(self._sequence_to_sympy(expr, symbols_dict)[0])
                expr = []
            else:
                expr.append(seq)

        if get_np_func:
            # note: t (time) is not considered
            funcs = [sy.lambdify(x_symbols, expr, 'numpy') for expr in exprs] # arity of np_func = d
            return exprs, funcs
        return exprs 

    def _get_sympy_op_args(self, expr: sy.Expr):
        for op,name in self.sympy_operators.items():
            if not isinstance(expr, (sy.Add, sy.Pow)) and isinstance(expr, op):
                return name, list(expr.args)
        
        if isinstance(expr, sy.Add):
            # subtraction
            # check 4 cases: (+-)x(+-)y
            args = list(expr.args)
            flag = False
            if (not isinstance(args[0], sy.Pow)
                and any([a == -1 for a in args[0].args])): # -x
                flag = True
                args[0] *= -1 # x
            if (not isinstance(args[1], sy.Pow)
                and any([a == -1 for a in args[1].args])): # -y
                args[1:] = [-a for a in args[1:]] # y
                if flag: # -x-y: sub(-x, y)
                    return 'sub', [-args[0]] + args[1:]
                else: # x-y: sub(x, y)
                    return 'sub', [args[0]] + args[1:]
            if flag: # -x+y: sub(-x, -y)
                return 'sub', [-args[0]] + [-a for a in args[1:]]
            # addition, x+y
            return 'add', list(expr.args)

        if isinstance(expr, sy.Pow):
            # inverse
            if expr.args[1] == -1:
                return 'inv', [expr.args[0]]
            # square
            if expr.args[1] == 2:
                return 'pow2', [expr.args[0]]
            # cube
            if expr.args[1] == 3:
                return 'pow3', [expr.args[0]]
            # sqrt
            if expr.args[1] == 1/2:
                return 'sqrt', [expr.args[0]]
            # pow
            return 'pow', list(expr.args)
        
        # unknown
        raise ValueError(f'invalid operator/value: {str(type(expr))}')
    
    def _sympy_to_sequence(self, expr: sy.Expr):
        if isinstance(expr, sy.Symbol):
            return [str(expr)]
        if isinstance(expr, (sy.Float, sy.Integer, sy.Rational, type(sy.pi))):
            # return [str(x)]
            return [str(round(float(expr), 4))]
        
        op,args = self._get_sympy_op_args(expr)
        if len(args) > 2:
            _op = 'add' if op == 'sub' else op
            _args = [args[0]] + [self.operators[_op](*args[1:])]
        else:
            _args = args
        args = sum([self._sympy_to_sequence(a) for a in _args], [])
        return [op] + args
    
    def sympy_to_sequence(self, exprs: List[sy.Expr]):
        "Returns a list of prefix strings from a list of sympy expressions."
        if isinstance(exprs, sy.Expr):
            exprs = [exprs]
        
        sequence = []
        for expr in exprs:
            sequence.extend(self._sympy_to_sequence(expr))
            sequence.append(SEP_TOKEN)
        return sequence[:-1] # ignore the last SEP_TOKEN

    def simplify_sympy(self, exprs: List[sy.Expr], timeout_sec: int=3):
        "Simplifies a list of sympy expressions within a timeout."
        if isinstance(exprs, sy.Expr):
            exprs = [exprs]

        exprs = exprs[:]
        arguments = []
        for i in range(len(exprs)):

            with timeout(timeout_sec):
                expr = exprs[i]
                exprs[i] = sy.nsimplify(expr)
 
            args = sorted([sym.name for sym in exprs[i].free_symbols])
            arguments.extend(args if args else ['<null>'])
            args.append(SEP_TOKEN)
        return exprs, arguments[:-1] # ignore the last SEP_TOKEN
            
    def simplify_sequence(self, sequence: List[str], timeout_sec: int=3, d: int=None):
        "Simplifies a list of prefix strings within a timeout."
        SymbolicExpression.check_sequence(sequence)

        exprs = self.sequence_to_sympy(sequence, False, d)
        exprs,arguments = self.simplify_sympy(exprs, timeout_sec)
        return self.sympy_to_sequence(exprs), arguments

    def get_readable_sequence(self, sequence: List[str], d: int=None):
        exprs = self.sequence_to_sympy(sequence[:], False, d)
        strs = []
        for i,expr in enumerate(exprs):
            s = str(expr).replace('**', '^').replace('*', ' * ').replace('^', '**')
            # rewrite power notation
            p = re.findall(r'x\d+\*\*', s)
            for pi in p:
                s = s.replace(pi, '(' + pi.rstrip('*') + ')**')
            
            # round float to 4 decimals
            p = re.findall(r'\d+.\d{4,}', s)
            for pi in p:
                s = s.replace(pi, str(round(float(pi), 4)))
            strs.append(f"dx{i+1}/dt = " + s)
        return strs

    def print_readable_sequence(self, sequence: List[str], d: int=None):
        strs = self.get_readable_sequence(sequence, d)
        for s in strs:
            print(s)