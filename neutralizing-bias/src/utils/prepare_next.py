import re
import spacy
import argparse
from typing import Tuple
from pytorch_pretrained_bert.tokenization import BertTokenizer

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)

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


def tokenize(s: str) -> str:
    """BERT-tokenize a given string.
    """
    global TOKENIZER
    tok_list = TOKENIZER.tokenize(s.strip())
    return " ".join(tok_list)


def detokenize(s: str) -> str:
    """De-tokenize a given string.
    """
    s = s.replace(" ##", "")
    s = re.sub(r'\s([.,:;?!)/])', r'\1', s)
    s = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', s)
    s = s.replace("( ", "(")
    s = s.replace(" ' ", "'")
    s = s.replace(" \\' ", "'")
    s = s.replace(" \\'", "'")
    return s


# Read input
pred_seqs = []
gold_seqs = []
pred_seqs_tok = []
gold_seqs_tok = []
with open(ARGS.input, "r") as f:
    for line in f:
        if re.match(r'^PRED SEQ:', line):
            pred_seq_tok = line.split("\t")[1][3:-2]
            pred_seq_tok = pred_seq_tok.replace("\\", "")
            pred_seqs_tok.append(pred_seq_tok)
            pred_seq = detokenize(pred_seq_tok)
            pred_seqs.append(pred_seq)

        if re.match(r'^GOLD SEQ:', line):
            gold_seq_tok = line.split("\t")[1][3:-2]
            gold_seq_tok = gold_seq_tok.replace("\\", "")
            gold_seqs_tok.append(gold_seq_tok)
            gold_seq = detokenize(gold_seq_tok)
            gold_seqs.append(gold_seq)

# Write output
with open(ARGS.output, "w") as f:
    for i , (ps_tok, gs_tok, ps, gs) in enumerate(zip(pred_seqs_tok, gold_seqs_tok, pred_seqs, gold_seqs)):        
        if ps_tok.startswith("##"):
            continue

        pos, dep = get_pos_dep(ps_tok)

        # if pos and dep:
        f.write("\t".join(["0", ps_tok, gs_tok, ps, gs, pos, dep]) + "\n")
