import argparse
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="plot scores")

parser.add_argument("file_path", type=str, default="", help="scores")

args = parser.parse_args()

scores = [float(score) for score in open(args.file_path)]

plt.plot(range(1, len(scores) + 1), scores)
plt.xlabel("beam width")
plt.ylabel("BLEU score")
plt.savefig(args.file_path.replace(".txt", ".png"))
