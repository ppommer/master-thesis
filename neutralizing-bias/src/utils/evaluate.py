import re
import sys; sys.path.append('.')
import numpy as np
from seq2seq.utils import get_bleu


for i in range(10):
    hits = []
    pred_data = []
    gold_data = []

    for line in open("inference/concurrent_many/results_concurrent_" + str(i + 1) + ".txt"):
        if re.match(r'^PRED SEQ:', line):
            pred = line.split("\t")[1][3:-2]
            pred_data.append(pred.split(" "))

        if re.match(r'^GOLD SEQ:', line):
            gold = line.split("\t")[1][3:-2]
            gold_data.append(gold.split(" "))

    for pred, gold in zip(pred_data, gold_data):
        hits.append(1) if pred == gold else hits.append(0)

    assert len(pred_data) == len(gold_data)
    
    with open("utils/stats.txt", "a") as f:
        f.write("{:.2f}\n".format(np.mean(hits) * 100))
        
        # for pred, gold in zip(pred_data, gold_data):
        #     f.write("{:>6.2f}\n".format(get_bleu(pred, gold)))
