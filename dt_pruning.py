# coding=utf-8
import dt
from tree import Node


def simple_pruning(T, a, cost_func=dt.H):
    """

    :param T: Tree
    :param a: a >= 0
    :param cost_func: H(D), Gini(D), square_error(D)
    :return:
    """
    c_ta = cost_func(T.sample) * len(T.sample)
    if not T.children:
        return c_ta

    c_tb = a * (len(T.children) - 1)
    for child in T.children.values():
        c_tb += simple_pruning(child, a)

    if c_ta <= c_tb:
        # prunning
        T.children = {}
        T.val = dt.max_cnt(T.sample)
        return c_ta
    return c_tb


def rep(T):
    pass


def pep(T):
    pass


def mep(T):
    pass


def ccp(T, cost_func=dt.Gini):
    gts = []
    _cal_pruning_a(T, cost_func, gts)
    # remove the root
    gts = gts[:-1]
    T.gt = None
    Tn = [T]
    gts.sort()
    for i, a in enumerate(gts):
        Tk = _pruning_a(Tn[i], a)
        Tn.append(Tk)
    return Tn


def _pruning_a(T, a):
    # copy node
    node = Node(val=T.val, feature=T.feature, sample=T.sample, gt=T.gt)
    if T.gt == a:
        # pruning
        node.val = dt.max_cnt(T.sample)
        node.feature = None
        return node
    children = {}
    if T.children:
        for key, child in T.children.items():
            children[key] = _pruning_a(child, a)
    if children:
        node.children = children
    return node


def _cal_pruning_a(T, cost_func=dt.Gini, gts=[]):
    c_r = cost_func(T.sample) * len(T.sample)
    if not T.children:
        return c_r
    c_R = 0
    for child in T.children.values():
        c_R += _cal_pruning_a(child, cost_func, gts)
    gt = (c_r - c_R) / (len(T.children) - 1)
    T.gt = gt
    gts.append(gt)
    return c_R


def ebp(T):
    pass


def cvp(T):
    pass
