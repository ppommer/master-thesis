import re
import spacy
import argparse
from typing import Tuple
from pytorch_pretrained_bert.tokenization import BertTokenizer

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--gold", type=str)

# Define global variables
ARGS = parser.parse_args()
NLP = spacy.load("en_core_web_sm") # run "python -m spacy download en_core_web_sm" to initially download the model
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="cache")

assert ARGS.input, "Need to specify input arg!"
assert ARGS.output, "Need to specify output arg!"

# Define functions
def get_pos_dep(s: str) -> Tuple[str, str]:
    """Get POS and dependency tags for a given string.
    """
    toks = s.split()

    def words_from_toks(toks):
        words = []
        word_indices = []
        for i, tok in enumerate(toks):
            if tok.startswith('##'):
                words[-1] += tok.replace('##', '')
                word_indices[-1].append(i)
            else:
                words.append(tok)
                word_indices.append([i])
        return words, word_indices

    out_pos, out_dep = [], []
    words, word_indices = words_from_toks(toks)
    analysis = NLP(" ".join(words))

    if len(analysis) != len(words):
        return None, None

    for analysis_tok, idx in zip(analysis, word_indices):
        out_pos += [analysis_tok.pos_] * len(idx)
        out_dep += [analysis_tok.dep_] * len(idx)

    assert len(out_pos) == len(out_dep) == len(toks)

    return " ".join(out_pos), " ".join(out_dep)


def tokenize(s: str):
    """BERT-tokenize a given string.
    """
    global TOKENIZER
    tok_list = TOKENIZER.tokenize(s.strip())
    return " ".join(tok_list)


# Read gold seqs and toks
gold_seqs = []
gold_seqs_tok = []
with open(ARGS.gold, "r") as gold_file:
    for line in gold_file:
        gold_seqs.append(line.split("\t")[4])
        gold_seqs_tok.append(line.split("\t")[2])

# Read pred seqs and toks
pred_seqs = []
pred_seqs_tok = []
with open(ARGS.input, "r") as in_file:
    for line in in_file:
        pred_seq = line.strip()
        pred_seq_tok = tokenize(pred_seq)
        pred_seqs.append(pred_seq)
        pred_seqs_tok.append(pred_seq_tok)

# Write output
with open(ARGS.output, "w") as out_file:
    for ps_tok, gs_tok, ps, gs in zip(pred_seqs_tok, gold_seqs_tok, pred_seqs, gold_seqs):
        pos, dep = get_pos_dep(ps_tok)
        out_file.write("\t".join(["0", ps_tok, gs_tok, ps, gs, pos, dep]) + "\n")
