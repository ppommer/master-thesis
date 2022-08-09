import os
import re


source_path = "../../neutralizing-bias/src/data"
target_path = "../data/WNC"

train_set = set()
dev_set = set()
test_set = set()

neutral_test_set = set()


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
for line in open(os.path.join(source_path, "WNC/biased.full"), "r"):
    train_set.add(clean(line.split("\t")[4].rstrip()))

for line in open(os.path.join(source_path, "WNC/biased.word.dev"), "r"):
    dev_set.add(clean(line.split("\t")[4].rstrip()))

for line in open(os.path.join(source_path, "WNC/biased.word.test"), "r"):
    test_set.add(clean(line.split("\t")[4].rstrip()))

for line in open(os.path.join(source_path, "WNC/single_neutral_test.txt"), "r"):
    neutral_test_set.add(line.rstrip())

for line in open(os.path.join(source_path, "WNC/multi_neutral_test.txt"), "r"):
    neutral_test_set.add(line.rstrip())

print("Read {} train sentences".format(len(train_set)))
print("Read {} dev sentences".format(len(dev_set)))
print("Read {} test sentences".format(len(test_set)))
print()

train_set -= dev_set
train_set -= test_set
train_set -= neutral_test_set
dev_set -= neutral_test_set

print("Overlap train-dev: {}".format(len(train_set & dev_set)))
print("Overlap train-test: {}".format(len(train_set & test_set)))
print("Overlap dev-test: {}".format(len(dev_set & test_set)))
print("Overlap train-neutral_test: {}".format(len(train_set & neutral_test_set)))
print("Overlap dev_neutral_test: {}".format(len(dev_set & neutral_test_set)))
print()

print("{} train sentences are left after removing test sentences".format(len(train_set)))
print("{} dev sentences are left after removing test sentences".format(len(dev_set)))


# Write sentences to files
with open(os.path.join(target_path, "WNC_full/train.txt"), "w") as train_file:
    with open(os.path.join(target_path, "WNC_full/train.label"), "w") as train_label_file:
        train_file.write("\n".join(train_set) + "\n")
        train_label_file.write("neutral\n" * len(train_set))

with open(os.path.join(target_path, "WNC_full/dev.txt"), "w") as dev_file:
    with open(os.path.join(target_path, "WNC_full/dev.label"), "w") as dev_label_file:
        dev_file.write("\n".join(dev_set) + "\n")
        dev_label_file.write("neutral\n" * len(dev_set))

with open(os.path.join(target_path, "WNC_full/test.txt"), "w") as test_file:
    with open(os.path.join(target_path, "WNC_full/test.label"), "w") as test_label_file:
        test_file.write("\n".join(test_set) + "\n")
        test_label_file.write("neutral\n" * len(test_set))
