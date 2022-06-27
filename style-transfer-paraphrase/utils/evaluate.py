import argparse
import numpy as np
import sys; sys.path.append('.')
from tqdm import tqdm
from style_paraphrase.evaluation.similarity.test_sim import find_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--pred_data', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--gold_data', type=str, default="datasets/WNC/test_neutral.txt")
parser.add_argument('--in_data', type=str, default="datasets/WNC/test_biased.txt")
parser.add_argument('--batch_size', type=int, default=16)
ARGS = parser.parse_args()

sim_scores_gold = []
sim_scores_in = []

with open(ARGS.pred_data, "r") as f:
    pred_data = f.read().strip().split("\n")

with open(ARGS.gold_data, "r") as f:
    gold_data = f.read().strip().split("\n")

with open(ARGS.in_data, "r") as f:
    in_data = f.read().strip().split("\n")

assert len(pred_data) == len(gold_data) == len(in_data)

for i in range(0, len(pred_data), ARGS.batch_size):
    sim_scores_gold.extend(
        find_similarity(
            pred_data[i:i + ARGS.batch_size], 
            gold_data[i:i + ARGS.batch_size]
        )
    )

    sim_scores_in.extend(
        find_similarity(
            pred_data[i:i + ARGS.batch_size], 
            in_data[i:i + ARGS.batch_size]
        )
    )

with open(ARGS.output, "w") as f:
    f.write("==================\n")
    f.write("SIM GOLD: {:>8,.4f}\n".format(np.mean(sim_scores_gold)))
    f.write("SIM IN:   {:>8,.4f}\n".format(np.mean(sim_scores_in)))
    f.write("==================\n")
    f.write(" " * 7 + "GOLD |  IN\n")

    for i, (ssg, ssi) in tqdm(enumerate(zip(sim_scores_gold, sim_scores_in)), desc="Evaluation"):
        f.write("{} - {:>4.2f} | {:>4.2f}\n".format(str(i + 1).zfill(4), ssg, ssi))

    f.write("==================\n")