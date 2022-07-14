import numpy as np
import random


files = [
    "stats_modular_single.txt",
    "stats_modular_multi.txt",
    "stats_modular_modular_single.txt",
    "stats_modular_modular_multi.txt",
    "stats_modular_concurrent_single.txt",
    "stats_modular_concurrent_multi.txt",
    "stats_concurrent_single.txt",
    "stats_concurrent_multi.txt",
    "stats_concurrent_concurrent_single.txt",
    "stats_concurrent_concurrent_multi.txt",
    "stats_concurrent_modular_single.txt",
    "stats_concurrent_modular_multi.txt",
    "stats_strap_word_single_0.txt",
    "stats_strap_word_multi_0.txt",
    "stats_strap_full_single_0.txt",
    "stats_strap_full_multi_0.txt",
    "stats_strap_large_single_0.txt",
    "stats_strap_large_multi_0.txt",
]

print("Statistical significance with bootstrap resampling and a 95% confidence level:")

for file in files:
    sample_means = []
    boot_means = []

    # Parse file and extract scores
    for line in open(file, "r"):
        if "-" in line:
            if "strap" in file:
                score = float(line.split("-")[1].split("|")[0].strip())
            else:
                score = float(line.split("-")[1].strip())
            sample_means.append(score)

    # Perform bootstrap resampling and save means
    for _ in range(100):
        boot_sample = np.random.choice(sample_means, replace=True, size=10)
        boot_means.append(boot_sample)

    boot_means_np = np.array(boot_means)
    boot_mean = np.mean(boot_means_np)

    # Calculate the 95% confidence interval (CI)
    percentile = np.percentile(boot_means_np, [2.5, 97.5])

    # Check if the mean of the bootstrap-resampled means lies within the CI
    result = boot_mean > percentile[0] and boot_mean < percentile[1]

    print("{}: {}".format(file, result))
