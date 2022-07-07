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

import argparse
from googleapiclient import discovery
from time import sleep
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--pred_data', type=str)
parser.add_argument('--in_data', type=str)
parser.add_argument('--gold_data', type=str)
parser.add_argument('--output', type=str)
ARGS = parser.parse_args()
API_KEY = "AIzaSyAgMnXDNEnCktdqwOznAp5SScv6B6-g5E8"
flavors = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]

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

with open(ARGS.input, "r") as in_file:
    with open(ARGS.output, "w") as out_file:
        for line in tqdm(in_file, desc="Evaluate toxicity..."):
            analyze_request["comment"]["text"] = line.replace("\n", "")
            response = client.comments().analyze(body=analyze_request).execute()
            
            #TODO: write output in stats style [0000 - tox_in | tox_pred | tox_gold]

            toxicity_score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

            # for flavor in flavors:
            #     out_file.write("{}: {:.2f}".format(flavor, response["attributeScores"][flavor]["summaryScore"]["value"]))

            sleep(1)
