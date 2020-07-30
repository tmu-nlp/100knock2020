import matplotlib.pyplot as plt
import re

def read_score(filename):
    with open(filename) as f:
        x = f.readlines()[1]
        x = re.search(r'(?<=BLEU4 = )\d*\.\d*(?=,)', x)
        return float(x.group())

xs = range(1, 21)
ys = [read_score(f'knock95.{x}.score') for x in xs]
plt.plot(xs, ys)
plt.show()