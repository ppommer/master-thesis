"""
The style transfer models were trained with p = 0.0, but feel free to experiment with this slider if the 
paraphrases are too close to the input. Increasing the p value results in more diverse paraphrases at the 
expense of content preservation. Refer to Holtzman et al. 2019 for more details.

Increasing the p value results in more diverse stylistic properties, but at the expense of content preservation. 
Experiment with this slider to get the desired output, you will get different output samples on each run for 
larger p values. Some styles seem to benefit from higher p values like 0.6 and 0.9 (see Table 15 in our paper 
for more details).
"""

import argparse
import sys
import torch

from style_paraphrase.inference_utils import GPT2Generator

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="/home/ppommer/repos/master-thesis/style-transfer-paraphrase/models/paraphraser_gpt2_large", type=str)
parser.add_argument('--top_p_value', default=0.6, type=float)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir, upper_length="same_5")

print("Loading WNC model...")
wnc = GPT2Generator("/home/ppommer/repos/master-thesis/style-transfer-paraphrase/style_paraphrase/models/WNC_neutral_large/checkpoint-153096")

input_sentence = input("\nEnter your sentence, q to quit: ")
top_p = float(input("Enter a top_p value between 0 and 1: "))

while top_p < 0.0 or top_p > 1.0:
    top_p = float(input("Enter a top_p value between 0 and 1: "))

while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
    print("\nInput: {}".format(input_sentence))
    
    # greedy
    paraphraser.modify_p(top_p=0.0)
    wnc.modify_p(top_p=0.0)
    intermediate_paraphrase = paraphraser.generate(input_sentence)
    transferred_output = wnc.generate(intermediate_paraphrase)
    print("\nGreedy Sample\n{}".format(transferred_output))

    # top_p
    wnc.modify_p(top_p=top_p)
    intermediate_paraphrases, _ = paraphraser.generate_batch([input_sentence, input_sentence, input_sentence])

    i = 0
    for ip in intermediate_paraphrases:
        transferred_outputs, _ = wnc.generate_batch([ip, ip, ip])
        for to in transferred_outputs:
            i += 1
            print("\nSample #{}".format(i))
            print("Intermediate Paraphrase: {}".format(ip))
            print("Transferred Output: {}".format(to))

    # input
    input_sentence = input("\nEnter your sentence, q to quit: ")

print("Exiting...")
