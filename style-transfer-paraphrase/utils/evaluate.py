import argparse
import math
import numpy as np
import sys; sys.path.append('.')
from collections import Counter
from tqdm import tqdm
from style_paraphrase.evaluation.similarity.test_sim import find_similarity
from pytorch_pretrained_bert.tokenization import BertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--pred_data', type=str, default="inference/strap_large/output_strap_large_multi_9.txt")
parser.add_argument('--output', type=str, default="inference/strap_large/stats_strap_large_multi_9.txt")
parser.add_argument('--gold_data', type=str, default="data/WNC/multi_neutral_test.txt")
parser.add_argument('--in_data', type=str, default="data/WNC/multi_biased_test.txt")
parser.add_argument('--batch_size', type=int, default=16)
ARGS = parser.parse_args()

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="cache")


def tokenize(s: str) -> str:
    """BERT-tokenize a given string.
    """
    global TOKENIZER
    tok_list = TOKENIZER.tokenize(s.strip())
    return " ".join(tok_list)


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


sim_scores_gold = []
sim_scores_in = []
bleu_scores_gold = []
bleu_scores_in = []
hits_gold = []
hits_in = []

with open(ARGS.pred_data, "r") as f:
    pred_data = f.read().strip().split("\n")

with open(ARGS.gold_data, "r") as f:
    gold_data = f.read().strip().split("\n")

with open(ARGS.in_data, "r") as f:
    in_data = f.read().strip().split("\n")

assert len(pred_data) == len(gold_data) == len(in_data)

for i in tqdm(range(0, len(pred_data), ARGS.batch_size), desc="Calculate similarity..."):
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

pred_data = [tokenize(x).split(" ") for x in pred_data]
gold_data = [tokenize(x).split(" ") for x in gold_data]
in_data = [tokenize(x).split(" ") for x in in_data]

for p, g, i in tqdm(zip(pred_data, gold_data, in_data), desc="Calculate BLEU and accuracy...", total=len(pred_data)):
    bleu_scores_gold.append(get_bleu(p, g))
    bleu_scores_in.append(get_bleu(p, i))
    hits_gold.append(1) if p == g else hits_gold.append(0)
    hits_in.append(1) if p == i else hits_in.append(0)

with open(ARGS.output, "w") as f:
    f.write("=" * 46 + "\n")
    f.write("BLEU GOLD: {:>35,.2f}\n".format(get_bleu(pred_data, gold_data)))
    f.write("BLEU IN:   {:>35,.2f}\n".format(get_bleu(pred_data, in_data)))
    f.write("ACC GOLD:  {:>35,.4f}\n".format(np.mean(hits_gold)))
    f.write("ACC IN:    {:>35,.4f}\n".format(np.mean(hits_in)))
    f.write("SIM GOLD:  {:>35,.4f}\n".format(np.mean(sim_scores_gold)))
    f.write("SIM IN:    {:>35,.4f}\n".format(np.mean(sim_scores_in)))
    f.write("=" * 46 + "\n")
    f.write(" " * 7 + "BLEU GOLD | BLEU IN | SIM GOLD | SIM IN\n")

    for i, (bsg, bsi, ssg, ssi) in tqdm(enumerate(zip(bleu_scores_gold, bleu_scores_in, sim_scores_gold, sim_scores_in)), desc="Write output...", total=len(bleu_scores_gold)):
        f.write("{} - {:>9.2f} | {:>7.2f} | {:>8.2f} | {:>6.2f}\n".format(str(i + 1).zfill(4), bsg, bsi, ssg, ssi))

    f.write("=" * 46 + "\n")