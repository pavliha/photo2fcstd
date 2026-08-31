import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
strata = {r["part"]: r for r in json.load(open(os.path.join(ROOT, "data", "axis_strata.json")))}
pvc = json.load(open(os.path.join(ROOT, "runs", "photo_vs_carve", "scores.json")))
tab = np.array(pvc["table"], float)
parts = [p for p in pvc["parts"] if p in strata][:len(tab)]

keep = [(strata[p], tab[i]) for i, p in enumerate(parts) if p in strata]
circ = np.array([s["circle"] for s, _ in keep])
px = np.array([s["px"] for s, _ in keep])
photo = np.array([t[0] for _, t in keep])
learned = np.array([t[1] for _, t in keep])
thin = np.array([t[2] for _, t in keep])

print("%d parts with photos, a carve and a stratum\n" % len(keep))
print("  %-34s %5s %7s %9s %9s" % ("stratum", "n", "photo", "thinnest", "learned"))
for name, sel in (("everything", np.ones(len(keep), bool)),
                  ("traced as a single circle", circ),
                  ("a real outline (not one circle)", ~circ),
                  ("real outline, face >= 12 px", (~circ) & (px >= 12))):
    if sel.sum():
        print("  %-34s %5d %7.3f %9.3f %9.3f" % (name, sel.sum(), photo[sel].mean(),
                                                 thin[sel].mean(), learned[sel].mean()))
        print("  %-34s %5s %7s %9s %9s" % ("", "", "", "", "carve wins %.0f%%" % (100 * np.mean(learned[sel] > photo[sel]))))
