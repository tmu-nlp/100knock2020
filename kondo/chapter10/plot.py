import sys
import matplotlib.pyplot as plt

path = sys.argv[1]
scores = []

with open(path) as f:
    for line in f:
        score = float(line.strip())
        scores.append(score)

fig = plt.figure()

labels = range(len(scores))

plt.plot(labels, scores)
plt.show()
fig.savefig("img.png")
