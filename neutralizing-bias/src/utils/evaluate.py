import re
import argparse
import sys; sys.path.append('.')
from seq2seq.utils import get_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str)
parser.add_argument('--output', type=str)
ARGS = parser.parse_args()

pred_data = []
gold_data = []

with open(ARGS.results, "r") as f:
    for line in f:
        if re.match(r'^PRED SEQ:', line):
            pred_data.append(line.split("\t")[1][3:-2].split(" "))

        if re.match(r'^GOLD SEQ:', line):
            gold_data.append(line.split("\t")[1][3:-2].split(" "))

assert len(pred_data) == len(gold_data)

with open(ARGS.output, "w") as f:
    for pred, gold in zip(pred_data, gold_data):
        f.write("{:>6.2f}\n".format(get_bleu(pred, gold)))
