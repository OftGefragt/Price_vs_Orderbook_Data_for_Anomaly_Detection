import numpy as np
import random as rn


def c_factor(n):
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


class iForest:
    def __init__(self, X, ntrees, sample_size, limit=None, ExtensionLevel=0):
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.exlevel = ExtensionLevel
        self.CheckExtensionLevel()
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))
        self.c = c_factor(self.sample)
        for i in range(self.ntrees):
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(iTree(X_p, 0, self.limit, exlevel=self.exlevel))

    def CheckExtensionLevel(self):
        dim = self.X.shape[1]
        if self.exlevel < 0 or self.exlevel > dim-1:
            raise Exception(f"Extension level must be between 0 and {dim-1}.")

    def compute_paths(self, X_in=None):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i], self.Trees[j]).path
            Eh = h_temp / self.ntrees
            S[i] = 2.0**(-Eh / self.c)
        return S


class Node:
    def __init__(self, X, n, p, e, left, right, node_type=''):
        self.e = e
        self.size = len(X)
        self.X = X
        self.n = n
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type


class iTree:
    def __init__(self, X, e, l, exlevel=0):
        self.exlevel = exlevel
        self.e = e
        self.X = X
        self.size = len(X)
        self.dim = self.X.shape[1]
        self.Q = np.arange(self.dim, dtype='int')
        self.l = l
        self.p = None
        self.n = None
        self.exnodes = 0
        self.root = self.make_tree(X, e, l)

    def make_tree(self, X, e, l):
        self.e = e
        if e >= l or len(X) <= 1:
            self.exnodes += 1
            return Node(X, self.n, self.p, e, None, None, 'exNode')
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        idxs = np.random.choice(range(self.dim), self.dim - self.exlevel - 1, replace=False)
        self.n = np.random.normal(0, 1, self.dim)
        self.n[idxs] = 0
        self.p = np.random.uniform(mins, maxs)
        w = (X - self.p).dot(self.n) < 0
        return Node(
            X, self.n, self.p, e,
            left=self.make_tree(X[w], e + 1, l),
            right=self.make_tree(X[~w], e + 1, l),
            node_type='inNode'
        )


class PathFactor:
    def __init__(self, x, itree):
        self.path_list = []
        self.x = x
        self.e = 0
        self.path = self.find_path(itree.root)

    def find_path(self, T):
        if T.ntype == 'exNode':
            if T.size <= 1:
                return self.e
            else:
                self.e += c_factor(T.size)
                return self.e
        p = T.p
        n = T.n
        self.e += 1
        if (self.x - p).dot(n) < 0:
            self.path_list.append('L')
            return self.find_path(T.left)
        else:
            self.path_list.append('R')
            return self.find_path(T.right)


def all_branches(node, current=None, branches=None):
    if current is None:
        current = []
    if branches is None:
        branches = []
    current = current[:node.e]
    if node.ntype == 'inNode':
        current.append('L')
        all_branches(node.left, current=current, branches=branches)
        current = current[:-1]
        current.append('R')
        all_branches(node.right, current=current, branches=branches)
    else:
        branches.append(current)
    return branches
