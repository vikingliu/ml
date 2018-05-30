# coding=utf-8
import math
import sys
from collections import defaultdict

from tree import Node


def H(D):
    """
    计算数据集D的熵,
    :param D: [[x11,x12,x13...x1n,y],.... [xm1,xm2,xm3...xmn,y]]
    :return: -sum(pk * log(pk)), k=1,2,3..  = len(set(y))
    """
    p = defaultdict(int)
    for row in D:
        y = row[-1]
        p[y] += 1

    h = 0
    for y in p.keys():
        p[y] /= len(D) * 1.0
        h += p[y] * math.log(p[y], 2)

    return h * -1


def Gini(D):
    """
    计算gini指数
    :param D: [[x11,x12,x13...x1n,y],.... [xm1,xm2,xm3...xmn,y]]
    :return: 1 - sum(pk * pk), k=1,2,3..  = len(set(y))
    """
    p = defaultdict(int)
    for row in D:
        y = row[-1]
        p[y] += 1

    sum_pk = 0
    for y in p.keys():
        p[y] /= len(D) * 1.0
        sum_pk += p[y] * p[y]

    return 1 - sum_pk


def g(D, A, alg='g'):
    """
    计算信息增益或者增益率
    :param D: [[x11,x12,x13...x1n,y],.... [xm1,xm2,xm3...xmn,y]]
    :param A: split feature A index
    :param alg: alg in ['g', 'gr']
    :return: (g or gr, split Di)
    """
    Di = split(D, A)
    HDA = 0
    HAD = 0
    for i, di in Di.items():
        p = 1.0 * len(di) / len(D)
        HDA += p * H(di)
        HAD += p
    if alg == 'gr':
        return (H(D) - HDA) / HAD, Di
    else:
        return H(D) - HDA, Di


def gr(D, A):
    return g(D, A, 'gr')


def min_Gini(D, A):
    """
     计算特征 A所有属性的Gini, 返回最小的Gini和划分
    :param D: [[x11,x12,x13...x1n,y],.... [xm1,xm2,xm3...xmn,y]]
    :param A: split feature A index
    :return: min Gini for all feature A's values
    """
    Di = split(D, A)
    min_gDA = sys.maxint
    split_D = {}

    for i, D1 in Di.items():
        pk = 1.0 * len(D1) / len(D)
        D2 = []
        for j in Di.keys():
            if j != i:
                D2.extend(Di[j])
        gDA = pk * Gini(D1) + 1.0 * len(D2) / len(D) * Gini(D2)
        if gDA < min_gDA:
            min_gDA = gDA
            split_D = {i: D1}
            if D2:
                split_D['#other'] = D2

    return min_gDA, split_D


def square_error(D):
    c = sum([row[-1] for row in D]) * 1.0 / len(D)
    s = sum([math.pow((row[-1] - c), 2) for row in D])
    return s


def min_square_error(D, A):
    D = sorted(D, key=lambda x: x[A])
    n = len(D)
    min_error = sys.maxint
    split_D = {}
    for s in range(1, n):
        R1 = D[0:s]
        R2 = D[s:n]
        error = square_error(R1) + square_error(R2)
        if error < min_error:
            min_error = error
            split_D = {'<=%s' % D[s - 1][0]: R1, '>%s' % D[s - 1][0]: R2}
    return min_error, split_D


def split(D, A):
    Di = defaultdict(list)
    for item in D:
        Di[item[A]].append(item)
    return Di


def max_cnt(D):
    stats = defaultdict(int)
    for y in D:
        if type(y) is list:
            y = y[-1]
        stats[y] += 1
    stats = sorted(stats.items(), key=lambda x: x[1])
    return stats[0][0]


def train(D, features, best_split_func, has_split_A=[], alg='id3', e=0.01, depth=1, max_depth=sys.maxint, min_sample=1):
    split_feature = None
    Di = None
    rst = []
    d = [item[-1] for item in D]
    if len(set(d)) != 1 and depth < max_depth and len(d) > min_sample:
        for A in range(len(D[0]) - 1):
            if A in has_split_A:
                continue
            gDA, gDi = best_split_func(D, A)
            rst.append((gDA, gDi, A))

        if alg in ['id3', 'c45'] and rst:
            gDA, Di, split_feature = max(rst)
            if gDA < e:
                split_feature = None

        elif alg in ['cart', 'cart_r'] and rst:
            gDA, Di, split_feature = min(rst)

    if split_feature is None:
        if alg in ['cart_r']:
            val = sum(d) * 1.0 / len(d)
        else:
            val = max_cnt(d)

        return Node(val, sample=D)

    children = {}
    for v, di in Di.items():
        child = train(di, features, best_split_func, has_split_A + [split_feature], alg, e, depth + 1, max_depth,
                      min_sample)
        if child:
            children[v] = child

    return Node(features[split_feature], split_feature, children, D)


def classify(model, data):
    if model.children:
        val = data[model.feature]
        sub_model = model.children.get(val, None)
        sub_model = sub_model if sub_model else model.children['#other']
        return classify(sub_model, data)
    else:
        return model.val


def regression(model, data):
    if model.children:
        val = data[model.feature]
        for split_point, sub_model in model.children.items():
            if '<=' in split_point and val <= float(split_point.replace('<=', '')):
                break
        return classify(sub_model, data)
    else:
        return model.val


def id3(D, features):
    return train(D, features, g)


def c45(D, features):
    return train(D, features, gr)


def cart(D, features):
    return train(D, features, min_Gini, [], 'cart')


def cart_r(D, features):
    return train(D, features, min_square_error, [], 'cart_r')


if __name__ == '__main__':
    data = [
        [u'青年', u'否', u'否', u'一般', u'否'],
        [u'青年', u'否', u'否', u'好', u'否'],
        [u'青年', u'是', u'否', u'好', u'是'],
        [u'青年', u'是', u'是', u'一般', u'是'],
        [u'青年', u'否', u'否', u'一般', u'否'],
        [u'中年', u'否', u'否', u'一般', u'否'],
        [u'中年', u'否', u'否', u'好', u'否'],
        [u'中年', u'是', u'是', u'好', u'是'],
        [u'中年', u'否', u'是', u'非常好', u'是'],
        [u'中年', u'否', u'是', u'非常好', u'是'],
        [u'老年', u'否', u'是', u'非常好', u'是'],
        [u'老年', u'否', u'是', u'好', u'是'],
        [u'老年', u'是', u'否', u'好', u'是'],
        [u'老年', u'是', u'否', u'非常好', u'是'],
        [u'老年', u'否', u'否', u'一般', u'否'],
    ]
    data1 = [[1, 4.5], [2, 4.75], [3, 4.91], [4, 5.34], [5, 5.80], [6, 7.05], [7, 7.90], [8, 8.23], [9, 8.70],
             [10, 9.00]]
    f = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况']
    f1 = ['x']
    model = cart(data, f)
    # print classify(model, [u'老年', u'否', u'是', u'非常好'])
    #print regression(model, [5.5])
    import dt_pruning
    dt_pruning.ccp(model, Gini)

    model.show()
