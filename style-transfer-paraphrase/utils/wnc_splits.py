import os
import re
from tqdm import tqdm


source_path = "../../neutralizing-bias/src/bias_data"
target_path = "../datasets/WNC"

sentences = set()
test_sentences = set()

train_list = []
dev_list = []
test_list = []


def clean(s: str) -> str:
    """Clean a given string.
    """
    s = re.sub(r'\s([.,:;?!)/])', r'\1', s)
    s = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', s)
    s = s.replace("( ", "(")
    s = s.replace(" ' ", "'")
    s = s.replace(" \\' ", "'")
    s = s.replace(" \\'", "'")
    return s


# Read sentences
with open(os.path.join(source_path, "WNC/biased.full"), "r") as in_file:
    for line in tqdm(in_file, desc="Adding biased.full..."):
        sentences.add(clean(line.split("\t")[4].replace("\n", "")))

with open(os.path.join(source_path, "WNC/neutral"), "r") as in_file:
    for line in tqdm(in_file, desc="Adding neutral..."):
        sentences.add(clean(line.split("\t")[4].replace("\n", "")))

print("{} sentences prepared".format(len(sentences)))

# Remove test sentences from set
with open(os.path.join(source_path, "WNC_edit/multiword_neutral_test.txt"), "r") as in_file:
    for line in in_file:
        test_list.append(line.replace("\n", ""))
        test_sentences.add(line.replace("\n", ""))

with open(os.path.join(source_path, "WNC_edit/singleword_neutral_test.txt"), "r") as in_file:
    for line in in_file:
        test_list.append(line.replace("\n", ""))
        test_sentences.add(line.replace("\n", ""))

print("Removing {} test sentences".format(len(test_sentences)))
sentences -= test_sentences
print("{} sentences remaining".format(len(sentences)))

# Add sentences to lists
for sentence in tqdm(sentences, desc="Creating sentence lists..."):
    if len(dev_list) < 0.05 * len(sentences):
        dev_list.append(sentence)
    else:
        train_list.append(sentence)

print("Train list contains {} sentences".format(len(train_list)))
print("Dev list contains {} sentences".format(len(dev_list)))
print("Test list contains {} sentences".format(len(test_list)))

# Write sentences to files
with open(os.path.join(target_path, "WNC_large/train.txt"), "w", encoding="utf8") as train_file:
    with open(os.path.join(target_path, "WNC_large/train.label"), "w", encoding="utf8") as train_label_file:
        train_file.write("\n".join(train_list) + "\n")
        train_label_file.write("neutral\n" * len(train_list))

with open(os.path.join(target_path, "WNC_large/dev.txt"), "w", encoding="utf8") as dev_file:
    with open(os.path.join(target_path, "WNC_large/dev.label"), "w", encoding="utf8") as dev_label_file:
        dev_file.write("\n".join(dev_list) + "\n")
        dev_label_file.write("neutral\n" * len(dev_list))

with open(os.path.join(target_path, "WNC_large/test.txt"), "w", encoding="utf8") as test_file:
    with open(os.path.join(target_path, "WNC_large/test.label"), "w", encoding="utf8") as test_label_file:
        test_file.write("\n".join(test_list) + "\n")
        test_label_file.write("neutral\n" * len(test_list))
