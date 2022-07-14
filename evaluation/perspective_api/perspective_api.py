"""
Perspective API
By default, we set a quota limit to an average of 1 query per second (QPS) for all Perspective projects. This limit should be enough for testing the API and for working in developer environments. If you're running a production website, you may need to request a quota increase.

Attributes/Flavors:
- TOXICITY: A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.
- SEVERE_TOXICITY: A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.
- IDENTITY_ATTACK: Negative or hateful comments targeting someone because of their identity.
- INSULT: Insulting, inflammatory, or negative comment towards a person or a group of people.
- PROFANITY: Swear words, curse words, or other obscene or profane language.
- THREAT: Describes an intention to inflict pain, injury, or violence against an individual or group.
"""
import os
import numpy as np
from googleapiclient import discovery
from time import sleep
from tqdm import tqdm

API_KEY = "AIzaSyAgMnXDNEnCktdqwOznAp5SScv6B6-g5E8"
flavors = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]

files = [
    "single_biased_test.txt",
    "multi_biased_test.txt",
    "output_modular_single.txt",
    "output_modular_multi.txt",
    "output_modular_2.txt",
    "output_modular_multi_2.txt",
    "output_modular_concurrent_single.txt",
    "output_modular_concurrent_multi.txt",
    "output_concurrent_single.txt",
    "output_concurrent_multi.txt",
    "output_concurrent_2.txt",
    "output_concurrent_multi_2.txt",
    "output_concurrent_modular_single.txt",
    "output_concurrent_modular_multi.txt",
    "output_strap_word_single_0.txt",
    "output_strap_word_multi_0.txt",
    "output_strap_full_single_0.txt",
    "output_strap_full_multi_0.txt",
    "output_strap_large_single_0.txt",
    "output_strap_large_multi_0.txt",
    "single_neutral_test.txt",
    "multi_neutral_test.txt",
]

folders = [
    "source_single",
    "source_multi",
    "modular_single",
    "modular_multi",
    "modular_modular_single",
    "modular_modular_multi",
    "modular_concurrent_single",
    "modular_concurrent_multi",
    "concurrent_single",
    "concurrent_multi",
    "concurrent_concurrent_single",
    "concurrent_concurrent_multi",
    "concurrent_modular_single",
    "concurrent_modular_multi",
    "strap_word_single",
    "strap_word_multi",
    "strap_full_single",
    "strap_full_multi",
    "strap_large_single",
    "strap_large_multi",
    "target_single",
    "target_multi",
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
