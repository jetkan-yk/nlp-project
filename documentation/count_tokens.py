import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

tokens = []

for dirpath, dirnames, filenames in os.walk("data"):
    if dirnames != ["info-units", "triples"]:
        continue

    for filename in filenames:
        # load article
        if filename.endswith("-Stanza-out.txt"):
            with open(os.path.join(dirpath, filename)) as f:
                token = 0
                article = f.read().splitlines()
                for sent in article:
                    token += len(sent.split())
                tokens.append(token)

tokens = np.array(tokens)

plt.hist(
    tokens,
    bins=range(1500, 9900, 100),
    density=True,
    cumulative=True,
    color="grey",
)

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel("Number of tokens")
plt.ylabel("Cumulative number of documents")
plt.show()
