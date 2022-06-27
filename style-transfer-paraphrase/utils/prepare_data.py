import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="/home/ppommer/repos/master-thesis/neutralizing-bias/src/bias_data/WNC/biased.word.test")
parser.add_argument('--output_bias', type=str, default="datasets/WNC/test_biased.txt")
parser.add_argument('--output_neutral', type=str, default="datasets/WNC/test_neutral.txt")
ARGS = parser.parse_args()

with open(ARGS.input, "r") as in_file:
    with (open(ARGS.output_bias, "w")) as out_file_bias:
        with (open(ARGS.output_neutral, "w")) as out_file_neutral:
            for line in in_file:
                out_file_bias.write(line.split("\t")[3] + "\n")
                out_file_neutral.write(line.split("\t")[4] + "\n")
