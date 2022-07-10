import os
import numpy as np
from googleapiclient import discovery
from time import sleep
from tqdm import tqdm

API_KEY = "AIzaSyAgMnXDNEnCktdqwOznAp5SScv6B6-g5E8"
flavors = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]

files = [
    "ibc_biased.txt",
    "news_biased.txt",
    "speeches_biased.txt",
    "output_modular_ibc.txt",
    "output_modular_news.txt",
    "output_modular_speeches.txt",
    "output_concurrent_ibc.txt",
    "output_concurrent_news.txt",
    "output_concurrent_speeches.txt",
    "output_strap_ibc.txt",
    "output_strap_news.txt",
    "output_strap_speeches.txt",
    "output_modular_concurrent_ibc.txt",
    "output_modular_concurrent_news.txt",
    "output_modular_concurrent_speeches.txt",
    "output_concurrent_modular_ibc.txt",
    "output_concurrent_modular_news.txt",
    "output_concurrent_modular_speeches.txt",
]

folders = [
    "source_ibc",
    "source_news",
    "source_speeches",
    "modular_ibc",
    "modular_news",
    "modular_speeches",
    "concurrent_ibc",
    "concurrent_news",
    "concurrent_speeches",
    "strap_ibc",
    "strap_news",
    "strap_speeches",
    "modular_concurrent_ibc",
    "modular_concurrent_news",
    "modular_concurrent_speeches",
    "concurrent_modular_ibc",
    "concurrent_modular_news",
    "concurrent_modular_speeches",
]

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl=
    "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

analyze_request = {
    "comment": {
        "text": ""
    },
    "languages": ["en"],
    "requestedAttributes": {
        "TOXICITY": {},
        "SEVERE_TOXICITY": {},
        "IDENTITY_ATTACK": {},
        "INSULT": {},
        "PROFANITY": {},
        "THREAT": {},
    },
    "spanAnnotations": True # return scores at per-sentence-level
}

for file, folder in zip(files, folders):
    scores = {}

    for flavor in flavors:
        scores[flavor] = []

    for line in tqdm(open(os.path.join("data", file), "r").readlines(), desc="Evaluating toxicity for {}...".format(file)):
        if line == "\n":
            continue

        analyze_request["comment"]["text"] = line.replace("\n", "")
        response = client.comments().analyze(body=analyze_request).execute()

        for flavor in flavors:
            scores[flavor].append(response["attributeScores"][flavor]["summaryScore"]["value"])
        
        sleep(1)

    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(os.path.join(folder, "summary.txt"), "w") as f:
        for flavor in flavors:
            f.write("{}: {:.4f}\n".format(flavor, np.mean(scores[flavor])))

    for flavor in flavors:
        with open(os.path.join(folder, flavor + ".txt"), "w") as f:
            for score in scores[flavor]:
                f.write("{:.4f}".format(score) + "\n")
