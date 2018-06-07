# coding=utf-8
import math
import sys

from decision_tree import max_cnt
from ml import ML


class AdaBoost(ML):
    def __init__(self, g, max_m=100, e=0.01):
        self.m = max_m
        self.g = g
        self.e = e
        self.model = None

    def train(self, data):
        n = len(data)
        dm = [1.0 / n] * n
        fs = []
        pre_em = sys.maxint
        for i in range(self.m):
            em, am, gm, dm = self._train_g(data, dm, g)
            print em
            fs.append((am, gm))
            g.model = None
            if em < self.e or pre_em - em < 0.01:
                break
            pre_em = em
        self.model = fs
        return fs

    def predict(self, test):
        if not self.model:
            return None
        rst = []
        for f in self.model:
            self.g.model = f[1]
            cls = f[0] * self.g.predict(test)
            rst.append(cls)
        return sum(rst)

    def _train_g(self, data, dm, g):
        gm = g.train(data, dm)
        em = g.error
        am = 0.5 * math.log((1 - em) / em)
        n = len(data)
        zm = sum([dm[i] * math.exp(-1 * am * data[i][-1] * g.predict(data[i])) for i in range(n)])
        dm_1 = [0] * n
        for i in range(n):
            dm_1[i] = dm[i] * math.exp(-1 * am * data[i][-1] * g.predict(data[i]))
            dm_1[i] /= zm

        return em, am, gm, dm_1

    def error(self, data, dm, g):
        e = 0
        for i, row in enumerate(data):
            y = g.predict(row)
            if y != row[-1]:
                e += dm[i]
        return e


class WeakCls(ML):
    def __init__(self):
        ML.__init__(self, 'weak')
        self.model = None
        self.splited = []
        self.error = 0

    def train(self, data, w=None):
        return self.min_split_error(data, 0, w)

    def predict(self, test):
        for split_v, val in self.model:
            if test[0] <= split_v:
                return val
        return val

    def cal_error(self, D, w=None):
        if not w:
            w = [1] * len(D)
        c = max_cnt(D)
        e = 0
        for i, row in enumerate(D):
            if row[-1] != c:
                e += w[i]
        return e

    def min_split_error(self, D, A, w=None):
        D = sorted(D, key=lambda x: x[A])
        n = len(D)
        min_error = sys.maxint
        split_D = None
        split_v = None
        for s in range(1, n):
            if s in self.splited:
                continue
            R1 = D[0:s]
            R2 = D[s:n]
            error = self.cal_error(R1, w) + self.cal_error(R2, w)
            if error < min_error:
                min_error = error
                split_v = s
                split_D = [(D[s - 1][0], max_cnt(R1)), (D[s - 1][0], max_cnt(R2))]
        self.model = split_D
        self.splited.append(split_v)
        self.error = min_error
        return self.model


if __name__ == '__main__':
    data = [
        [0, 1, 3, -1],
        [0, 3, 1, -1],
        [1, 2, 2, -1],
        [1, 1, 3, -1],
        [1, 2, 3, -1],
        [0, 1, 2, -1],
        [1, 1, 2, 1],
        [1, 1, 1, 1],
        [1, 3, 1, -1],
        [0, 2, 1, -1]
    ]
    features = ['身体', '业务', '潜力']

    data = [
        [0, 1],
        [1, 1],
        [2, 1],
        [3, -1],
        [4, -1],
        [5, -1],
        [6, 1],
        [7, 1],
        [8, 1],
        [9, -1],
    ]
    features = ['x']
    g = WeakCls()
    ada = AdaBoost(g)
    ada.train(data)
    print ada.predict([1, 3, 2])
