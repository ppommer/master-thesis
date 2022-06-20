import re
import spacy
import argparse
from typing import Tuple
from pytorch_pretrained_bert.tokenization import BertTokenizer

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")

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


# Read input and write output
with open(ARGS.input, "r") as in_file:
    with open(ARGS.output, "w") as out_file:
        for line in in_file:
            s = line.strip()
            tok = tokenize(s)
            pos, dep = get_pos_dep(tok)
            out_file.write("0\t" + (tok + "\t") * 2 + (s + "\t") * 2 + pos + "\t" + dep + "\n")
