import argparse
import math
import numpy as np
import sys; sys.path.append('.')
from collections import Counter
from tqdm import tqdm
from style_paraphrase.evaluation.similarity.test_sim import find_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default="inference/stats_multiword_baseline.txt")
parser.add_argument('--gold_data', type=str, default="datasets/WNC/multiword_neutral_test.txt")
parser.add_argument('--in_data', type=str, default="datasets/WNC/multiword_biased_test.txt")
parser.add_argument('--batch_size', type=int, default=16)
ARGS = parser.parse_args()


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


sim_scores = []
bleu_scores = []

with open(ARGS.gold_data, "r") as f:
    gold_data = f.read().strip().split("\n")

with open(ARGS.in_data, "r") as f:
    in_data = f.read().strip().split("\n")

assert len(gold_data) == len(in_data)

for i in tqdm(range(0, len(gold_data), ARGS.batch_size), desc="Calculate similarity..."):
    sim_scores.extend(
        find_similarity(
            gold_data[i:i + ARGS.batch_size], 
            in_data[i:i + ARGS.batch_size]
        )
    )

for g, i in tqdm(zip(gold_data, in_data), desc="Calculate bleu..."):
    bleu_scores.append(get_bleu(g.split(" "), i.split(" ")))

with open(ARGS.output, "w") as f:
    f.write("======================\n")
    f.write("BLEU: {:>16,.2f}\n".format(np.mean(bleu_scores)))
    f.write("SIM:  {:>16,.4f}\n".format(np.mean(sim_scores)))
    f.write("======================\n")
    f.write(" " * 9 + "BLEU | SIM\n")

    for i, (ss, bs) in tqdm(enumerate(zip(sim_scores, bleu_scores)), desc="Write output..."):
        f.write("{} - {:>6.2f} | {:>4.4f}\n".format(str(i + 1).zfill(4), bs, ss))

    f.write("======================\n")
