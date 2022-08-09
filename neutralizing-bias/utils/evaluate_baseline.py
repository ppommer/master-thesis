import sys; sys.path.append('.')
import numpy as np
from tqdm import tqdm
from seq2seq.utils import get_bleu

pred_data = []
gold_data = []
hits_source = []
hits_target = []

# Single-word
for line in open("data/WNC/single_test.txt", "r"):
    pred = line.split("\t")[1].rstrip()
    gold = line.split("\t")[2].rstrip()
    pred_data.append(pred.split(" "))
    gold_data.append(gold.split(" "))
    hits_source.append(1) if pred == gold else hits_source.append(0)
    hits_target.append(1) if gold == gold else hits_target.append(0)

assert len(pred_data) == len(gold_data) == len(hits_source)
assert len(gold_data) == len(gold_data) == len(hits_target)

with open("inference/stats_source_copy_single.txt", "w") as f:
    f.write("=============\n")
    f.write("BLEU: {:>7,.2f}\n".format(get_bleu(pred_data, gold_data)))
    f.write("ACC:  {:>7,.4f}\n".format(np.mean(hits_source)))
    f.write("=============\n")
    for i, (p, g) in tqdm(enumerate(zip(pred_data, gold_data)), desc="Write source_copy_single...", total=len(gold_data)):
        f.write("{} - {:>6.2f}\n".format(str(i + 1).zfill(4), get_bleu(p, g)))
    f.write("=============\n")
    
with open("inference/stats_target_copy_single.txt", "w") as f:
    f.write("=============\n")
    f.write("BLEU: {:>7,.2f}\n".format(get_bleu(gold_data, gold_data)))
    f.write("ACC:  {:>7,.4f}\n".format(np.mean(hits_target)))
    f.write("=============\n")
    for i, (p, g) in tqdm(enumerate(zip(gold_data, gold_data)), desc="Write target_copy_single...", total=len(gold_data)):
        f.write("{} - {:>6.2f}\n".format(str(i + 1).zfill(4), get_bleu(p, g)))
    f.write("=============\n")

# Multi-word
for line in open("data/WNC/multi_test.txt", "r"):
    pred = line.split("\t")[1].rstrip()
    gold = line.split("\t")[2].rstrip()
    pred_data.append(pred.split(" "))
    gold_data.append(gold.split(" "))
    hits_source.append(1) if pred == gold else hits_source.append(0)
    hits_target.append(1) if gold == gold else hits_target.append(0)

assert len(pred_data) == len(gold_data) == len(hits_source)
assert len(gold_data) == len(gold_data) == len(hits_target)

with open("inference/stats_source_copy_multi.txt", "w") as f:
    f.write("=============\n")
    f.write("BLEU: {:>7,.2f}\n".format(get_bleu(pred_data, gold_data)))
    f.write("ACC:  {:>7,.4f}\n".format(np.mean(hits_source)))
    f.write("=============\n")
    for i, (p, g) in tqdm(enumerate(zip(pred_data, gold_data)), desc="Write source_copy_multi...", total=len(gold_data)):
        f.write("{} - {:>6.2f}\n".format(str(i + 1).zfill(4), get_bleu(p, g)))
    f.write("=============\n")

with open("inference/stats_target_copy_multi.txt", "w") as f:
    f.write("=============\n")
    f.write("BLEU: {:>7,.2f}\n".format(get_bleu(gold_data, gold_data)))
    f.write("ACC:  {:>7,.4f}\n".format(np.mean(hits_target)))
    f.write("=============\n")
    for i, (p, g) in tqdm(enumerate(zip(gold_data, gold_data)), desc="Write target_copy_multi...", total=len(gold_data)):
        f.write("{} - {:>6.2f}\n".format(str(i + 1).zfill(4), get_bleu(p, g)))
    f.write("=============\n")
