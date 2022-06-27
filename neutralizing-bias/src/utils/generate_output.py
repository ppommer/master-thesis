import re
import argparse
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--html_file", type=str, default=None)
ARGS = parser.parse_args()

assert ARGS.in_file, "Need to specify input arg!"
assert ARGS.out_file, "Need to specify output arg!"


def gradient(x: float, start_col: Tuple[int]=(255, 255, 255), end_col: Tuple[int]=(250, 50, 50)) -> str:
    """ Returns the HEX code for a color at position x ∈ [0; 1] within a color gradient of start_col and end_col.
    """
    rgb = (
        int((1 - x) * start_col[0] + x * end_col[0]),
        int((1 - x) * start_col[1] + x * end_col[1]),
        int((1 - x) * start_col[2] + x * end_col[2]))
    return "#%02x%02x%02x" % rgb


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


# read input
results = []
with open(ARGS.in_file, "r") as f:
    lines = f.readlines()

    for line in lines:
        if re.match(r'^IN SEQ:', line):
            in_seq_tok = line.split("\t")[1][3:-2]
            in_seq = in_seq_tok.replace(" ##", "")
            in_seq = re.sub(r'\s([.,;?!"])', r'\1', in_seq)

        if re.match(r'^PRED SEQ:', line):
            pred_seq_tok = line.split("\t")[1][3:-2]
            pred_seq = pred_seq_tok.replace(" ##", "")
            pred_seq = re.sub(r'\s([.,;?!"])', r'\1', pred_seq)

        if re.match(r'^PRED DIST:', line):
            pred_dist = line.split("\t")[1]
            pred_dist = [float(x) for x in pred_dist[2:-2].split(", ")]

            results.append({
                "in_seq_tok": in_seq_tok,
                "pred_seq": pred_seq,
                "pred_dist": pred_dist,
            })

# write output
with open(ARGS.out_file, "w") as f:
    text = "\n".join([result["pred_seq"] for result in results]) + "\n"
    text = detokenize(text)
    f.write(text)

# write HTML output
if ARGS.html_file is not None:
    html_list = []
    tok_lists = [x["in_seq_tok"] for x in results]
    dist_lists = [x["pred_dist"]for x in results]

    for toks, dists in zip(tok_lists, dist_lists):
        html_string = "<div style='background-color:white;padding:10px;margin:-8px'>"

        for tok, dist in zip(toks.split(" "), dists):
            html_string += "<span style='color:black;background-color: " + gradient(dist) + "'>" + tok + "</span>" + " "

        html_string += "</div>"
        html_list.append(html_string)

    with open(ARGS.html_file, "w") as f:
        f.write("\n".join(html_list) + "\n")
