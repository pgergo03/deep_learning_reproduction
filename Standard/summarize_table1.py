import ast
import argparse
import glob
import re

import pandas as pd

ROWS = [("std", "Standard CNN"), ("bil", "+ Bil."), ("noise", "+ Noise"), ("nb", "+ Noise + Bil."), ("bn", "+ Bil. + Noise")]
COLS = ["Clean", "FGSM", "Linf", "EoT", "L2", "C&W"]
PAPER = {
    "std": [74.5, 3.5, 0.2, 0.2, 1.3, 0.6],
    "bil": [69.0, 10.0, 1.0, 1.2, 11.9, 0.5],
    "noise": [68.5, 22.8, 25.5, 12.0, 49.6, 43.0],
    "nb": [67.9, 33.9, 36.5, 18.9, 58.5, 47.2],
    "bn": [67.5, 30.8, 28.1, 15.1, 54.1, 41.8],
}
C_IDX = 4  # c = 1.8 in CW_adv's default c list


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
seed_suffix = f"_seed_{args.seed}" if args.seed is not None else ""


def last_list(path, prefix):
    lines = [l for l in open(path) if l.startswith(prefix)]
    return ast.literal_eval(re.sub(r"np\.float64\((.*?)\)", r"\1", lines[-1].split("=", 1)[1].strip()))


ours = {}
for tag, _ in ROWS:
    res = glob.glob(f"Results/Bil_*_{tag}{seed_suffix}")[0]
    clean = pd.read_excel(f"{res}/m.xlsx").iloc[0, 1]
    clean_sd = pd.read_excel(f"{res}/v.xlsx").iloc[0, 1]
    fgsm, linf, eot, l2, _bpda = [100 * a for a in last_list(f"logs/mult_{tag}{seed_suffix}.log", "Epsilon: 0.03")]
    cw = 100 * last_list(f"logs/cw_{tag}{seed_suffix}.log", "Epsilon: 1")[C_IDX]
    ours[tag] = ([clean, fgsm, linf, eot, l2, cw], clean_sd)

lines = ["| Method | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for tag, label in ROWS:
    vals, sd = ours[tag]
    cells = [f"{v:.1f} ({p:.1f})" for v, p in zip(vals, PAPER[tag])]
    cells[0] = f"{vals[0]:.1f}±{sd:.1f} ({PAPER[tag][0]:.1f})"
    lines.append(f"| {label} | " + " | ".join(cells) + " |")
for name, fn in [("linear gain", lambda d, i: d["bil"][i] + d["noise"][i] - 2 * d["std"][i]),
                 ("actual gain", lambda d, i: d["nb"][i] - d["std"][i])]:
    o = {k: v[0] for k, v in ours.items()}
    lines.append(f"| {name} | " + " | ".join(f"{fn(o, i):+.1f} ({fn(PAPER, i):+.1f})" for i in range(len(COLS))) + " |")

out = "Accuracy in %, ours (paper). Clean is mean±std over the last 10 epochs.\n\n" + "\n".join(lines) + "\n"
report_name = f"table1{seed_suffix}.md"
open(report_name, "w").write(out)
print(out)
