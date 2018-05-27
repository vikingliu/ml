# coding=utf-8
from collections import defaultdict


def bayes(data):
    py = defaultdict(int)
    px_y = defaultdict(int)
    xs = {}
    for row in data:
        y = row[-1]
        py[y] += 1
        for x in row[0:-1]:
            px_y['x=%s|y=%s' % (x, y)] += 1
            xs[x] = 1

    for y, n in py.items():
        for x in xs:
            px_y['x=%s|y=%s' % (x, y)] /= n * 1.0
        py[y] /= len(data) * 1.0

    return py, px_y


def predict(x, py, px_y):
    max_p = 0
    p_y = 0
    for y, p in py.items():
        for i in x:
            p *= px_y['x=%s|y=%s' % (i, y)]
        if p > max_p:
            max_p = p
            p_y = y
    return p_y


if __name__ == '__main__':
    data = [[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
            [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
            [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'L', 1], [3, 'L', -1]]
    py, px_y = bayes(data)
    x = [2, 'S']
    print predict(x, py, px_y)
