"""
Lines to change for CPU use:
- inference_utils.py: 53, 95, 96, 97
- utils.py: 15, 16, 52
"""

import argparse
import sys
import torch

from style_paraphrase.inference_utils import GPT2Generator

parser = argparse.ArgumentParser()
parser.add_argument('--diverse_paraphraser', default="models/paraphraser_gpt2_large", type=str)
parser.add_argument('--inverse_paraphraser', default="models/strap_full", type=str)
parser.add_argument('--top_p_value', default=0.6, type=float)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading diverse paraphraser...")
paraphraser = GPT2Generator(args.diverse_paraphraser, upper_length="same_5")

print("Loading inverse paraphraser...")
wnc = GPT2Generator(args.inverse_paraphraser)

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
    print("\nGreedy sample\n{}".format(transferred_output))

    # top_p
    wnc.modify_p(top_p=top_p)
    intermediate_paraphrases, _ = paraphraser.generate_batch([input_sentence, input_sentence, input_sentence])

    i = 0
    for ip in intermediate_paraphrases:
        transferred_outputs, _ = wnc.generate_batch([ip, ip, ip])
        for to in transferred_outputs:
            i += 1
            print("\nSample #{}".format(i))
            print("Intermediate paraphrase: {}".format(ip))
            print("Transferred output: {}".format(to))

    # input
    input_sentence = input("\nEnter your sentence, q to quit: ")

print("Exiting...")
