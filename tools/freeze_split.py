"""Split the photographed parts into a tuning set and a test set nobody looks at."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

TEST_FRACTION = 0.25
SEED = 20260901
TUNE = "data/tune_ids.txt"
TEST = "data/test_ids.txt"


def main():
    if os.path.exists(TEST):
        print("%s already exists; refusing to redraw a frozen split" % TEST)
        return
    parts = sorted(set(open("data/all_photographed_ids.txt").read().split()))
    groups = json.load(open("data/part_groups.json"))
    keys = sorted({groups.get(p, -1) for p in parts})
    rng = np.random.default_rng(SEED)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    held = set(shuffled[:int(round(TEST_FRACTION * len(shuffled)))])
    test = [p for p in parts if groups.get(p, -1) in held]
    tune = [p for p in parts if groups.get(p, -1) not in held]
    open(TUNE, "w").write("\n".join(tune) + "\n")
    open(TEST, "w").write("\n".join(test) + "\n")
    print("tuning %d parts, test %d parts, split over %d geometry groups (seed %d)"
          % (len(tune), len(test), len(keys), SEED))


if __name__ == "__main__":
    main()
