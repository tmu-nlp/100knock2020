import matplotlib.pyplot as plt
import sys
import re
outputs_dir = "./data/outputs"
fname = sys.argv[1]
beams = [i+1 for i in range(100)]
bleus = []
with open(f"{outputs_dir}/{fname}") as f:
    for line in f:
        if line[0] == 'N': continue
        bleu = re.sub(r'^BLEU4\s=\s(.+?),.+', r'\1', line)
        bleus.append(float(bleu))

plt.xlabel("beam size")
plt.ylabel("BLEU")
plt.plot(beams, bleus)
plt.savefig(f"./picture/{fname}.png")
#plt.show()