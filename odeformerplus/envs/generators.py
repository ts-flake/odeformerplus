from __future__ import annotations
from typing import List

import json, re
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings(action="error", category=np.ComplexWarning)

np.seterr(invalid='raise', divide='raise', over='raise')

CONST_TOKENS = ['<const>', '<const_mul>', '<const_add>', '<const_mul_global>', '<const_add_global>']
TRIGONOMETRIC_OPERATORS = ['sin', 'cos', 'tan']
EXTRA_OPERATORS = {'pow': 2} # not used to sample expressions, only for representation

from .symbolic import SymbolicExpression, SYMBOL_RE, EXTRA_SYMBOL_RE, SEP_TOKEN
from .utils import load_config, timeout, print_dataset_stats, suppress_output

config = load_config('odeformerplus/envs/config.yaml')

def _traverse(node, l, condition=(lambda node: True)):
    if node is None: return l
    if condition(node): l.append(node)
    l = _traverse(node.lchild, l, condition)
    l = _traverse(node.rchild, l, condition)
    return l


class ExpressionTree():
    params = config.expression_tree

    def __init__(self, value='', lchild=None, rchild=None, parent=None):
        self.value = value
        self.lchild = lchild
        self.rchild = rchild
        self.parent = parent
    
    @property
    def is_root(self): return self.parent is None
    @property
    def is_leaf(self): return self.lchild is None and self.rchild is None
    @property
    def is_symbol(self): return bool(re.fullmatch(SYMBOL_RE, self.value))
    @property
    def is_extra_symbol(self): return bool(re.fullmatch(EXTRA_SYMBOL_RE, self.value))
    @property
    def is_operator(self):
        if not self.value: return 0
        if self.value in self.params.unary_operators: return 1
        if self.value in self.params.binary_operators: return 2
        if self.value in EXTRA_OPERATORS: return EXTRA_OPERATORS[self.value] # arity
        return 0
    @property
    def is_constant(self):
        if self.value in CONST_TOKENS: return True
        try: float(self.value); return True
        except ValueError: return False

    def get_all_succ_unary_operators(self):
        condition = (lambda node: node.is_operator == 1)
        l = _traverse(self, [], condition)
        l = [node.value for node in l]
        return list(set(l))
    
    def get_all_prec_unary_operators(self):
        l = []
        node = self
        while node.parent is not None:
            node = node.parent
            if node.is_operator == 1:
                l.append(node.value)
        return list(set(l))
    
    def get_all_unary_operators(self):
        return list(set(self.get_all_succ_unary_operators() + self.get_all_prec_unary_operators()))

    @staticmethod
    def depth(node):
        if node is None: return 0
        ldepth = ExpressionTree.depth(node.lchild)
        rdepth = ExpressionTree.depth(node.rchild)
        return max(ldepth, rdepth) + 1
        
    @staticmethod
    def get_D(n: int):
        """
        Refer to the appendix C1, https://arxiv.org/abs/1912.01412
        
        D(e,n) represents the number of different binary subtrees that can
        be generated from e empty nodes with n internal nodes.
        
        D(0,n) = 0
        D(e,0) = 1
        D(e,n) = D(e-1,n) + D(e+1,n-1)

        Args:
            n: maximum number of binary operators.
        """
        e = 2*n+1
        D = np.zeros((e+1,n+1), dtype=int)
        D[0,:]  = 0
        D[1:,0] = 1
        for nj in range(1,n+1):
            for ei in range(1,e-nj+1):
                D[ei,nj] = D[ei-1,nj] + D[ei+1,nj-1]
        return D

    @staticmethod
    def get_probability_k(D: np.ndarray, e: int, n: int):
        probs = np.array([D[e-k+1, n-1] for k in range(e)])
        return probs / D[e,n]
    
    @classmethod
    def sample_operator(cls, rng, arity: int=2, exclude: list=[]):
        if arity == 1: # unary
            ops = cls.params.unary_operators.copy()
            p = cls.params.unary_unnormalized_probs.copy()
            if exclude:
                p_dict = dict(zip(ops, p))
                for e in exclude: ops.remove(e)
                p = [p_dict[op] for op in ops]
        else:
            ops = cls.params.binary_operators.copy()
            p = cls.params.binary_unnormalized_probs.copy()
            if exclude:
                p_dict = dict(zip(ops, p))
                for e in exclude: ops.remove(e)
                p = [p_dict[op] for op in ops]
        return rng.choice(ops, p=(np.array(p) / sum(p)))

    @classmethod
    def sample_symbol(cls, rng, d: int=None):
        if rng.random() < cls.params.symbol_t_prob:
            return 't'
        d = d if d is not None else cls.params.max_xdim
        assert d >= 1 and d <= cls.params.max_xdim
        return f'x{rng.choice(d)+1:d}'
        
    @classmethod
    def sample_number(cls, rng, l=None, u=None, integer_prob=None, loguniform=False,
                      include_negative=True, return_str=True, str_float_precision=4):
        if not l: l = cls.params.constant_range_l
        if not u: u = cls.params.constant_range_u
        assert l <= u, "lower bound should be smaller than upper bound"
        if loguniform:
            assert l > 0, "lower bound should be positive for loguniform distribution"
            n = 10**rng.uniform(np.log10(l), np.log10(u))
            if include_negative:
                s = rng.choice([-1., 1.]) # sign
                n = s * n
        else:
            n = rng.uniform(l, u)
        p = cls.params.constant_integer_prob if integer_prob is None else integer_prob
        if rng.random() < p:
            _n = round(n)
            if _n == 0: n = -1 if _n < 0 else 1 # avoid zero
            else: n = _n
        if return_str:
            return (f"%.{str_float_precision}f"%n) if type(n) != int else str(n)
        else:
            return n

    def add_unary_operator(self, rng, n: int=0):
        "Iteratively add `n` unary operators to the tree."
        def condition(node):
            return (ExpressionTree.depth(node) < self.params.max_subtree_depth
                    and (node.is_operator or node.is_symbol or node.is_extra_symbol))

        # iteratively add unary operator
        for _ in range(n):
            nodes = _traverse(self, [], condition)
            node = rng.choice(nodes)
            lchild= ExpressionTree(node.value, node.lchild, node.rchild, node)
            
            # avoid nesting unary operators
            # e.g. we do not allow the followings
            # sin(x + cos(x) + ...)
            # exp(x + exp(x) + ...)
            # log(x + log(x) + ...)
            exclude = []
            if self.params.avoid_nesting_unary_operators:
                _unary_operators = node.get_all_unary_operators()
                if any(op in _unary_operators for op in TRIGONOMETRIC_OPERATORS):
                    exclude += TRIGONOMETRIC_OPERATORS
                if 'exp' in _unary_operators:
                    exclude += ['exp']
                if 'log' in _unary_operators:
                    exclude += ['log']
    
            node.value = node.sample_operator(rng, 1, exclude)
            node.lchild = lchild
            node.rchild = None
        
    def add_affine(self, rng):
        "Add affine transformation to unary operators and variables."
        def condition(node):
            return ((node.is_operator == 1 and node.value != 'id')
                    or node.is_symbol
                    or node.is_extra_symbol
                    or node.is_root)
        
        # add affine transformations
        nodes = _traverse(self, [], condition)
        pa,pb = (self.params.constant_mul_prob,
                           self.params.constant_add_prob)
        for node in nodes:
            if rng.random() < pa:
                a = ExpressionTree('<const_mul>', parent=node)
                rchild = ExpressionTree(node.value, node.lchild, node.rchild, node)
                node.value = 'mul'
                node.lchild = a
                node.rchild = rchild
            if rng.random() < pb:
                b = ExpressionTree('<const_add>', parent=node)
                rchild = ExpressionTree(node.value, node.lchild, node.rchild, node)
                node.value = 'add'
                node.lchild = b
                node.rchild = rchild
    
    def add_global_affine(self, a: str=None, b: str=None):
        if a:
            rchild = ExpressionTree(self.value, self.lchild, self.rchild, parent=self)
            self.value = 'mul'
            self.lchild = ExpressionTree(a, parent=self)
            self.rchild = rchild
        if b:
            rchild = ExpressionTree(self.value, self.lchild, self.rchild, parent=self)
            self.value = 'add'
            self.lchild = ExpressionTree(b, parent=self)
            self.rchild = rchild
    
    def remove_constant(self):
        "Remove constant tokens from the tree."
        def condition(node):
            return (node.is_operator == 2 and (node.lchild.is_constant ^ node.rchild.is_constant))
        
        # remove all constant tokens including numbers
        nodes = _traverse(self, [], condition)
        for node in reversed(nodes): # update nodes from the tip to the root
            if node.lchild.is_constant:
                node.value = node.rchild.value
                node.lchild = node.rchild.lchild
                node.rchild = node.rchild.rchild

            elif node.rchild.is_constant:
                node.value = node.lchild.value
                node.lchild = node.lchild.lchild
                node.rchild = node.lchild.rchild
            
            # update parent
            if node.lchild: node.lchild.parent = node
            if node.rchild: node.rchild.parent = node
    
    @classmethod
    def sample_binary_tree(cls, rng, d: int=None, n: int=None):
        "Sample a binary tree with `n` binary operators."
        root = cls()
        root.parent = None
        tree = [root]
        if n is not None:
            assert cls.params.min_num_binary_operators <= n <= cls.params.max_num_binary_operators
        else:
            n = rng.choice(range(cls.params.min_num_binary_operators, cls.params.max_num_binary_operators+1))
        D = cls.get_D(n)
        e = 1
        start = 0
        while n > 0:
            p = cls.get_probability_k(D, e, n)
            k = rng.choice(range(e), p=p)
            node = tree[start + k]
            node.value = cls.sample_operator(rng, 2)
            node.lchild = cls(cls.sample_symbol(rng, d), parent=node)
            node.rchild = cls(cls.sample_symbol(rng, d), parent=node)
            tree.extend([node.lchild, node.rchild])
            start = start+k+1
            e = e-k+1
            n = n-1
        return root

    def sample_constant(self, rng):
        "Fill constant tokens with random numbers."
        def _fill(node):
            if node is None: return
            if node.is_constant:
                if node.value == '<const>':
                    node.value = self.sample_number(
                        rng,
                        self.params.constant_range_l,
                        self.params.constant_range_u,
                        self.params.constant_integer_prob,
                        self.params.constant_loguniform,
                        self.params.constant_include_negative,
                        return_str=True
                    )
                elif node.value == '<const_mul>':
                    node.value = self.sample_number(
                        rng,
                        self.params.constant_mul_range_l,
                        self.params.constant_mul_range_u,
                        self.params.constant_mul_integer_prob,
                        self.params.constant_mul_loguniform,
                        self.params.constant_mul_include_negative,
                        return_str=True
                    )
                elif node.value == '<const_add>':
                    node.value = self.sample_number(
                        rng,
                        self.params.constant_add_range_l,
                        self.params.constant_add_range_u,
                        self.params.constant_add_integer_prob,
                        self.params.constant_add_loguniform,
                        self.params.constant_add_include_negative,
                        return_str=True
                    )
                elif node.value == '<const_mul_global>':
                    node.value = self.sample_number(
                        rng,
                        self.params.constant_mul_global_range_l,
                        self.params.constant_mul_global_range_u,
                        self.params.constant_mul_global_integer_prob,
                        self.params.constant_mul_global_loguniform,
                        self.params.constant_mul_global_include_negative,
                        return_str=True
                    )
                elif node.value == '<const_add_global>':
                    node.value = self.sample_number(
                        rng,
                        self.params.constant_add_global_range_l,
                        self.params.constant_add_global_range_u,
                        self.params.constant_add_global_integer_prob,
                        self.params.constant_add_global_loguniform,
                        self.params.constant_add_global_include_negative,
                        return_str=True
                    )
                elif node.is_constant: pass
                else:
                    raise ValueError(f'Unknown constant token: {node.value}')
            _fill(node.lchild)
            _fill(node.rchild)
        _fill(self)

    def resample_symbol(self, rng, d: int=None):
        "Resample symbols in the tree."
        nodes = _traverse(self, [], (lambda node: node.is_symbol))
        for node in nodes:
            node.value = self.sample_symbol(rng, d)

    def permute_symbol(self, rng):
        "Permute symbols in the tree. E.g. x1,x2,x3 -> x3,x1,x2."
        condition = (lambda node: node.is_symbol)
        nodes = _traverse(self, [], condition)
        symbols = list(set([n.value for n in nodes]))
        permuted_symbols = symbols[:]
        rng.shuffle(permuted_symbols)
        for node in nodes:
            node.value = permuted_symbols[symbols.index(node.value)]
    
    def relabel_symbol(self):
        "Relabel symbols in the tree such that they always start from 1."
        condition = (lambda node: (node.is_symbol and 'x' in node.value))
        nodes = _traverse(self, [], condition)
        symbols = list(set([n.value for n in nodes]))
        symbols.sort()
        relabelled_symbols = [f'x{i}' for i in range(1, len(symbols)+1)]
        for node in nodes:
            node.value = relabelled_symbols[symbols.index(node.value)]

    def tree_to_prefix_sequence(self) -> List[str]:
        nodes = _traverse(self, [])
        return [node.value for node in nodes]

    @classmethod
    def prefix_sequence_to_tree(cls, x: List[str]) -> ExpressionTree:
        x = x[:]
        def _to_tree(x, parent=None):
            tree = cls()
            tree.value = x.pop(0)
            tree.parent = parent
            if not x: return tree
            if tree.is_operator == 2:
                tree.lchild = _to_tree(x, tree)
                tree.rchild = _to_tree(x, tree)
            elif tree.is_operator == 1:
                tree.lchild = _to_tree(x, tree)
                tree.rchild = None
            return tree
        return _to_tree(x)

class ExpressionGenerator():
    """
    Generator of a multi-dimensional system of expressions.
    """

    params = config.expression_generator

    def __init__(self, rng_seed):
        self.rng = np.random.default_rng(rng_seed)
    
    def sample_multi_expression(self, d: int=None):
        """
        Generate a `d`-dimensional system of expressions as a prefix sequence.

        E.g. [f1, SEP_TOKEN, f2], where SEP_TOKEN is used to
        denote the boundary, f1 and f2 are prefix strings.
        """
        if d is None: d = self.rng.choice(ExpressionTree.params.max_xdim)
        assert d >= 1 and d <= ExpressionTree.params.max_xdim

        def _sample_tree_skeleton(nu=None, nb=None, is_root=True):
            # number of unary operators
            if not nu:
                nu = self.rng.choice(range(ExpressionTree.params.min_num_unary_operators,
                                           ExpressionTree.params.max_num_unary_operators+1))
            # number of binary operators
            if not nb:
                nb = self.rng.choice(range(ExpressionTree.params.min_num_binary_operators,
                                           ExpressionTree.params.max_num_binary_operators+1))
            tree = ExpressionTree.sample_binary_tree(self.rng, d, nb)
            tree.add_unary_operator(self.rng, nu) # unary-binary tree
            if not is_root:
                tree.parent = ExpressionTree() # dummy parent, prevent global affine
            tree.add_affine(self.rng)
            return tree, nu, nb
        
        subtrees = {}
        trees = []
        # generate expressions with biases:
        # - having common subexpressions,
        # - same expression w/ different symbols,
        # - same expression w/ different constants

        # a list of candidate subexpressions
        subtree_candidates = []
        _count = 0
        while _count < self.params.num_subexpression_candidates:
            _skeleton,_nu,_nb = _sample_tree_skeleton(is_root=False)
            _depth = ExpressionTree.depth(_skeleton)
            if self.params.min_subtree_depth <= _depth <= self.params.max_subtree_depth:
                _count += 1

                _instance = deepcopy(_skeleton)
                _instance.sample_constant(self.rng)
                subtree_candidates.append(
                    {'skeleton': _skeleton,
                    'instance': _instance,
                    'nu': _nu,
                    'nb': _nb}
                )
        
        for _ in range(d):
            p = np.array(self.params.bias_type_unnormalized_probs)
            bias = self.rng.choice(self.params.bias_types, p=(p/p.sum()))
            assert bias in ['subexpression', 'symbol', 'permute', 'constant', 'independent'], 'Unknown bias type'

            # use subexpression
            if bias == 'subexpression':
                candidate = self.rng.choice(subtree_candidates)
                subtree = candidate[self.rng.choice(['skeleton', 'instance'])]
                # sample a tree with remaining numbers of operators
                nu = ExpressionTree.params.max_num_unary_operators - candidate['nu']
                nb = ExpressionTree.params.max_num_binary_operators - candidate['nb']
                if nb > 0:
                    nu = self.rng.choice(range(ExpressionTree.params.min_num_unary_operators, nu+1))
                    nb = self.rng.choice(range(ExpressionTree.params.min_num_binary_operators, nb+1))
                    subtree_name = f'g{len(subtrees)+1}'
                    tree,_,_ = _sample_tree_skeleton(nu, nb)

                    exclude = []
                    _unary_operators = subtree.get_all_unary_operators()
                    if any(op in _unary_operators for op in TRIGONOMETRIC_OPERATORS):
                        exclude += TRIGONOMETRIC_OPERATORS
                    if 'exp' in _unary_operators:
                        exclude += ['exp']
                    if 'log' in _unary_operators:
                        exclude += ['log']

                    def condition(node):
                        return node.value not in exclude and node.is_leaf
                    
                    nodes = _traverse(tree, [], condition)
                    if nodes:
                        node = self.rng.choice(nodes)
                        node.value = subtree_name # insert the subtree and treat it as a whole
                        tree = {
                            'tree': tree, 
                            'subtree': {
                                'name': subtree_name,
                                'tree': deepcopy(subtree)
                            }
                        }
                    else:
                        tree = {'tree': deepcopy(candidate['skeleton']), 'subtree': None}
                else:
                    tree = {'tree': deepcopy(candidate['skeleton']), 'subtree': None}    
            
            # use existing expression w/ different symbols
            elif len(trees) > 0 and bias == 'symbol':
                tree = deepcopy(self.rng.choice(trees))
                tree['tree'].resample_symbol(self.rng, d)
                if tree['subtree'] and self.rng.random() < 0.5:
                        tree['subtree']['tree'].resample_symbol(self.rng, d)
            
            # use existing expression w/ permuted symbols
            elif len(trees) > 0 and bias == 'permute':
                tree = deepcopy(self.rng.choice(trees))
                tree['tree'].permute_symbol(self.rng)
                if tree['subtree'] and self.rng.random() < 0.5:
                        tree['subtree']['tree'].permute_symbol(self.rng)
            
            # use existing expression w/ different constants
            elif len(trees) > 0 and bias == 'constant':
                tree = deepcopy(self.rng.choice(trees))
                tree['tree'].remove_constant()
                tree['tree'].add_affine(self.rng)                
                if tree['subtree'] and self.rng.random() < 0.5:
                        tree['subtree']['tree'].remove_constant()
                        tree['subtree']['tree'].parent = ExpressionTree() # dummy parent, prevent global affine
                        tree['subtree']['tree'].add_affine(self.rng)
            
            # indepent
            else:
                tree,_,_ = _sample_tree_skeleton()
                tree = {'tree': tree, 'subtree': None}

            trees.append(tree)
        
        # sample constants
        for tree in trees:
            tree['tree'].sample_constant(self.rng)
            # TODO: simplify tree
            sequence = tree['tree'].tree_to_prefix_sequence()
            sequence = SymbolicExpression().simplify_sequence(sequence, self.params.simplify_timeout_sec)[0]
            # tree['tree'] = ExpressionTree.prefix_sequence_to_tree(sequence)
            tree['sequence'] = sequence

            # sample constants and substitute subtrees
            if tree['subtree'] and (tree['subtree']['name'] in tree['sequence']):
                tree['subtree']['tree'].sample_constant(self.rng)
                # TODO: simplify subtree
                sequence = tree['subtree']['tree'].tree_to_prefix_sequence()
                sequence = SymbolicExpression().simplify_sequence(sequence, self.params.simplify_timeout_sec)[0]
                # tree['subtree']['tree'] = ExpressionTree.prefix_sequence_to_tree(sequence)
                tree['subtree']['sequence'] = sequence
                # substitute
                idx = tree['sequence'].index(tree['subtree']['name'])
                tree['sequence'] = tree['sequence'][:idx] + tree['subtree']['sequence'] + tree['sequence'][idx+1:]
                # tree['tree'] = ExpressionTree.prefix_sequence_to_tree(tree['sequence'])
        
            if self.params.simplify_all:
                tree['sequence'] = SymbolicExpression().simplify_sequence(tree['sequence'], self.params.simplify_timeout_sec)[0]
        
        # add global affine to all expressions
        global_const = []
        if self.rng.random() < self.params.constant_add_global_prob:
            b = ExpressionTree.sample_number(
                self.rng,
                self.params.constant_add_global_range_l,
                self.params.constant_add_global_range_u,
                self.params.constant_add_global_integer_prob,
                self.params.constant_add_global_loguniform,
                self.params.constant_add_global_include_negative,
                return_str=True
            )
            global_const += ['add', b]
        
        if self.rng.random() < self.params.constant_mul_global_prob:
            a = ExpressionTree.sample_number(
                self.rng,
                self.params.constant_mul_global_range_l,
                self.params.constant_mul_global_range_u,
                self.params.constant_mul_global_integer_prob,
                self.params.constant_mul_global_loguniform,
                self.params.constant_mul_global_include_negative,
                return_str=True
            )
            global_const += ['mul', a]

        # convert trees to prefix sequences
        res = sum([global_const + tree['sequence'] + [SEP_TOKEN] for tree in trees], [])[:-1] # ignore the last SEP_TOKEN
        del trees, subtrees, subtree_candidates
        return res

class DatasetGenerator():
    params = config.dataset_generator

    def __init__(self, rng_seed):
        self.rng = np.random.default_rng(rng_seed)
        self.expression_generator = ExpressionGenerator(rng_seed)
        self.symbolic_expression = SymbolicExpression()
    
    def _check_ode_sequence_length(self, sequence: List[str]):
        _seq = []
        for seq in sequence + [SEP_TOKEN]:
            if seq == SEP_TOKEN:
                if len(_seq) > self.params.max_sequence_len_per_dim:
                    return False
                _seq = []
            else:
                _seq.append(seq)
        return True
    
    def sample_ode_sequence(self, d: int=None, return_arguments: bool=False, debug: bool=False):
        "Return a `d`-dimensional system of ODEs as a prefix sequence."
        # we catch any invalid expressions
        # such as complex number in the try-except
        try:
            sequence = self.expression_generator.sample_multi_expression(d)
            if not self._check_ode_sequence_length(sequence):
                raise Exception('exceed max sequence length, expression too complex')
            arguments = []
            args = []
            for seq in sequence + [SEP_TOKEN]:
                if re.fullmatch(SYMBOL_RE, seq):
                    args.append(seq)
                if seq == SEP_TOKEN:
                    if not args:
                        raise Exception('discard constant function')
                    
                    arguments.extend(list(set(args)) + [SEP_TOKEN])
                    args.clear()
        except Exception as err:
            if debug:
                print(err)
            return self.sample_ode_sequence(d, return_arguments, debug)

        if return_arguments:
            return sequence, arguments[:-1]
        else:
            return sequence

    def get_ode_func_from_sequence(self, sequence: List[str]):
        _,funcs = self.symbolic_expression.sequence_to_sympy(sequence, get_np_func=True)
        def ode(t, x):
            xd = [f(*x) for f in funcs]
            return np.stack(xd)
        return ode

    # TODO: sample scheme
    def sample_x0_normal(self, d: int=1, n: int=1, r: float=1):
        "Sample `n` points from a `d`-dimensional Gaussian."
        return self.rng.normal(scale=r, size=(n,d))
    
    def sample_x0_uniform(self, d: int=1, n: int=1, r: float=1):
        "Sample `n` points uniformly from a `d`-dimensional hypersphere."
        u = self.rng.normal(0, 1, size=(n,d))
        norm = np.linalg.norm(u, axis=1, keepdims=True)
        r = self.rng.random((n,1))**(1.0 / d) * r
        return r * u / norm
    
    def _integrate(self, func, t_span, x0, t_eval=None, options: dict=None, check_x_range: bool=True, debug: bool=False):
        def value_event(t, x):
            return np.min(self.params.x_range - np.abs(x))
        value_event.terminal = True
        events = [value_event] if check_x_range else None

        if options is None:
            options = self.params.solver_options.to_dict()

        assert type(options) == dict, 'expect a dict `options`'
        
        # TODO: suppress lsoda stdout, stderr
        with suppress_output():
            res = solve_ivp(func, t_span, x0, t_eval=t_eval, events=events, **options)
        try:
            time = res.t   # (n,)
            traj = res.y # (d,n)
            traj = np.vstack([time, traj]).T # (n,d+1)
        except Exception as err:
            if debug:
                print('integrate failed')
                print(err)
                print(res.message)
            return None
        return traj
    
    def integrate_ode_func(self, func, x0, t_eval=None, num_pts: int=None, debug: bool=False):
        if num_pts is None:
            num_pts = self.rng.choice(range(self.params.min_num_pts_traj, self.params.max_num_pts_traj+1))
        if t_eval is None:
            t_eval = np.linspace(self.params.traj_time_range_l, self.params.traj_time_range_u, num_pts)

        try:
            with timeout(self.params.traj_timeout_sec):
                traj = self._integrate(func, (min(t_eval), max(t_eval)), x0, t_eval=t_eval, debug=debug) # (n,d+1)
                if traj is None: return None
                
                err = []
                if np.any(np.isnan(traj)): err.append('nan')
                if np.any(np.isinf(traj)): err.append('inf')
                if len(traj) != num_pts: err.append(f'not enough points, expect {num_pts} but got {len(traj)}')
                if np.any(np.iscomplex(traj)): err.append('complex value')
                if err:
                    if debug: print(';'.join(err))
                    return None
                
                # check x range
                if np.any(np.abs(traj[:,1:]) > self.params.x_range):
                    if debug: print('x out of range')
                    return None
                
                # discard converge
                # https://github.com/sdascoli/odeformer/blob/main/odeformer/envs/generators.py
                if self.rng.random() < self.params.discard_converge_traj_prob and num_pts > 10:
                    l = num_pts // 3
                    last = traj[-l:,1:] # (l,d)
                    if np.all(np.abs((np.max(last, axis=0) - np.min(last, axis=0)) / l) < self.params.check_converge_traj_value):
                        if debug: print('converge')
                        return None

        except Exception as err:
            if debug: print(err)
            return None
        return traj
    
    def _generate_data(self, num_lines: int, num_trajs: int, d: int=None, debug: bool=False):
        if d is None: d = self.rng.choice(range(1, ExpressionTree.params.max_xdim+1))
        
        # sample ode expression
        ode_sequence,ode_arguments = self.sample_ode_sequence(d, True, debug=debug)
        # ode func
        ode_func = self.get_ode_func_from_sequence(ode_sequence)

        # ---
        # generate a line field
        # ---
        if debug:
            print('-- sampled ODEs')
            self.symbolic_expression.print_readable_sequence(ode_sequence)
            print('\n-- trying to sample a line field...')
        
        num_lines_inner = int(num_lines * self.params.num_lines_inner_ratio)
        num_lines_outer = num_lines - num_lines_inner
        sz_inner = num_lines_inner * self.params.num_attempts_line_factor
        sz_outer = num_lines_outer * self.params.num_attempts_line_factor
        sz = sz_inner + sz_outer
        # sample t0
        _l = self.params.traj_time_range_l
        _u = self.params.traj_time_range_u
        dt = self.params.line_time_duration
        t0s = self.rng.uniform(_l, _u - dt, size=sz)
        # sample x0
        x0s_inner = self.sample_x0_uniform(d, sz_inner, self.params.x0_range_line_inner)
        x0s_outer = self.sample_x0_normal(d, sz_outer, self.params.x0_range_line_outer)

        lines = []
        # sample line inner
        for t0,x0 in zip(t0s[:sz_inner], x0s_inner):
            num_pts = self.params.num_pts_line
            t_eval = np.linspace(t0, t0+dt, num_pts)
            line = self.integrate_ode_func(ode_func, x0, t_eval=t_eval, num_pts=num_pts, debug=debug)
            
            if line is not None:
                # check line length not too large
                line_length = np.linalg.norm(line[1:,1:] - line[:-1,1:], axis=1).sum()
                if line_length < self.params.max_line_length:
                    lines.append(line)
            if len(lines) == num_lines_inner:
                break
        
        # sample line outer
        for t0,x0 in zip(t0s[sz_inner:], x0s_outer):
            num_pts = self.params.num_pts_line
            t_eval = np.linspace(t0, t0+dt, num_pts)
            line = self.integrate_ode_func(ode_func, x0, t_eval=t_eval, num_pts=num_pts, debug=debug)
            
            if line is not None:
                # check line length not too large
                line_length = np.linalg.norm(line[1:,1:] - line[:-1,1:], axis=1).sum()
                if line_length < self.params.max_line_length:
                    lines.append(line)
            if len(lines) == num_lines:
                break
        
        if debug:
            if len(lines) == num_lines:
                print('-- successfull')
            else:
                print('-- failed')

        if len(lines) != num_lines:
            return None

        # ---
        # generate trajectories
        # ---
        if debug:
            print('\n-- trying to sample trajectories...')
        
        sz = num_trajs*self.params.num_attempts_traj_factor
        # sample x0
        x0s = self.sample_x0_uniform(d, sz, self.params.x0_range_traj)

        trajs = []
        # sample trajectories
        for x0 in x0s:
            traj = self.integrate_ode_func(ode_func, x0, debug=debug)
            if traj is not None:
                trajs.append(traj)
            if len(trajs) == num_trajs:
                break
        
        if debug:
            if len(trajs) == num_trajs:
                print('-- successfull')
            else:
                print('-- failed')    
        
        if len(trajs) < self.params.min_num_trajs:
            return None
        
        return {
            'lines': lines,
            'trajectories': trajs,
            'dim': d,
            'ode': ode_sequence,
            'ode_arguments': ode_arguments
        }
    
    def generate_data(self, d: int=None, debug: bool=False):
        num_lines = self.params.num_lines
        num_trajs = self.rng.choice(range(self.params.min_num_trajs, self.params.max_num_trajs+1))

        try:
            with timeout(self.params.gen_data_timeout_sec):
                data = self._generate_data(num_lines, num_trajs, d, debug)
        except Exception as err:
            if debug: print('--', err)
            return None
        
        return data
    
    def generate_dataset(self, num_samples: int, dir_path: str, file_name: str='dataset.json', print_stats: bool=True):
        "Generate a dataset of `num_samples` samples."
        dataset = []
        dataset_stats = {
            'dimension': defaultdict(int),
            'num_trajs_per_dim': defaultdict(lambda: defaultdict(int)),
            'unary_operators': defaultdict(int),
            'binary_operators': defaultdict(int),
            'complexity_per_dim': [],
            'total_num_trajs': 0,
            'num_samples': num_samples
        }
        
        def _gen_data():
            while True:
                data = self.generate_data(debug=False)
                if data is not None:
                    return data

        with tqdm(total=num_samples) as pbar:
            for _ in range(num_samples):
                data = _gen_data()
                # record stats
                dataset_stats['dimension'][str(data['dim'])] += 1
                dataset_stats['num_trajs_per_dim'][str(data['dim'])][len(data['trajectories'])] += 1
                dataset_stats['total_num_trajs'] += len(data['trajectories'])
                
                for expr in data['ode']:
                    if expr in ExpressionTree.params.unary_operators:
                        dataset_stats['unary_operators'][expr] += 1
                    elif expr in ExpressionTree.params.binary_operators:
                        dataset_stats['binary_operators'][expr] += 1
                dataset_stats['complexity_per_dim'].append((len(data['ode']) - data['dim'] + 1) / data['dim'])

                # convert to list for json
                dataset.append({
                    'lines': [line.tolist() for line in data['lines']],
                    'trajectories': [traj.tolist() for traj in data['trajectories']],
                    'ode': data['ode'],
                    'ode_arguments': data['ode_arguments'],
                    'dim': int(data['dim'])
                })
                pbar.update(1)
        
        # save dataset
        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        file_name = file_name.split('.')[0]
        with open(dir_path / f'{file_name}.json', 'w') as f:
            json.dump(dataset, f)

        # save dataset stats   
        with open(dir_path / f'{file_name}_stats.json', 'w') as f:
            json.dump(dataset_stats, f)

        # print stats
        print(f"-- saved {num_samples} samples ({dataset_stats['total_num_trajs']} trajectories) to {dir_path/file_name}.json!")
        if print_stats:
            print_dataset_stats(dataset_stats)

