#coding=utf-8
import math
from collections import defaultdict
from tree import Node


def H(D):
    p = defaultdict(int)
    N = 0
    for item in D:
        y = item[-1]
        p[y] += 1
        N += 1

    h = 0
    for y in p.keys():
        p[y] /= N * 1.0
        h += p[y] * math.log(p[y], 2)

    return h * -1

def g(D, A, cal='g'):
    Di = split(D, A)
    HDA = 0
    HAD = 0
    for i, di in Di.items():
        p = 1.0 * len(di)/len(D)
        HDA += p * H(di)
        HAD += p
    if cal == 'gr':
        return (H(D) - HDA) / HAD
    else:
        return H(D) - HDA

def gr(D, A):
    return g(D, A, 'gr')

def split(D, A):
    Di = defaultdict(list)
    for item in D:
        Di[item[A]].append(item)
    return Di


def dt(D, features, f, split_A=[]):
    max_g = 0
    split_feature = None

    for A in range(len(D[0]) - 1):
        if A in split_A:
            continue
        gDA = f(D, A)
        if gDA > max_g:
            max_g = gDA
            split_feature = A

    if split_feature is None:
        return Node(D[0][-1])

    Di = split(D, split_feature)
    children = {}
    for v, di in Di.items():
        child = dt(di, features, f, split_A + [split_feature])
        if child:
            children[v] = child

    return Node(features[split_feature], children)

def id3(D, features):
    return dt(D, features, g)

def c45(D, features):
    return dt(D, features, gr)


def cart(D):
    pass


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
    features = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况']
    node = c45(data, features)

    node.show()









