import re
import sys; sys.path.append('.')
import numpy as np
from tqdm import tqdm
from seq2seq.utils import get_bleu


in_file = "inference/modular_concurrent/results_modular_concurrent_single.txt"
out_file = "inference/modular_concurrent/stats_modular_concurrent_single.txt"

hits = []
pred_data = []
gold_data = []

for line in open(in_file, "r"):
    if re.match(r'^PRED SEQ:', line):
        pred = line.split("\t")[1][3:-2]
        pred_data.append(pred.split(" "))

    if re.match(r'^GOLD SEQ:', line):
        gold = line.split("\t")[1][3:-2]
        gold_data.append(gold.split(" "))

    for pred, gold in zip(pred_data, gold_data):
        hits.append(1) if pred == gold else hits.append(0)

print(len(pred_data))
print(len(gold_data))
print(len(hits))

assert len(pred_data) == len(gold_data) == len(hits)

with open(out_file, "w") as f:
    f.write("=============\n")
    f.write("BLEU: {:>7,.2f}\n".format(get_bleu(pred_data, gold_data)))
    f.write("ACC:  {:>7,.4f}\n".format(np.mean(hits_source)))
    f.write("=============\n")
    for i, (p, g) in tqdm(enumerate(zip(pred_data, gold_data)), desc="Write output...", total=len(gold_data)):
        f.write("{} - {:>6.2f}\n".format(str(i + 1).zfill(4), get_bleu(p, g)))
    f.write("=============\n")
