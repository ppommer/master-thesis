import krippendorff
import numpy as np


bias_str = (
    "0	1	0",
    "-1	1	0",
    "1	2	1",
    "1	1	-2",
    "-1	1	0",
    "1	2	-2",
    "2	2	2",
    "2	2	-2",
    "2	2	1",
    "2	1	-1",
    "1	1	-1",
    "1	1	1",
    "1	2	-1",
    "2	2	-1",
    "1	2	0",
    "2	2	0",
    "2	2	2",
    "0	0	-1",
    "2	1	-2",
    "2	2	2",
)

fluency_str = (
    "1	1	1",
    "2	2	2",
    "2	2	2",
    "1	2	1",
    "1	1	1",
    "1	2	0",
    "-1	0	1",
    "1	2	2",
    "2	2	2",
    "2	2	2",
    "2	2	2",
    "1	0	-1",
    "1	1	0",
    "2	2	2",
    "1	1	1",
    "2	2	2",
    "2	1	-1",
    "0	-1	0",
    "2	2	2",
    "-1	1	-1",
)

meaning_str = (
    "1	0	0",
    "4	0	3",
    "3	0	3",
    "4	0	3",
    "1	1	1",
    "1	0	3",
    "3	0	3",
    "0	0	4",
    "1	0	3",
    "1	0	1",
    "1	0	3",
    "3	1	2",
    "1	0	2",
    "2	0	2",
    "4	1	4",
    "1	0	1",
    "3	0	4",
    "3	1	3",
    "0	0	1",
    "1	0	1",
)

bias_data = [[int(v) for v in coder.split()] for coder in bias_str]
fluency_data = [[int(v) for v in coder.split()] for coder in fluency_str]
meaning_data = [[int(v) for v in coder.split()] for coder in meaning_str]

print("Krippendorff's alpha (bias): {:.3f}".format(krippendorff.alpha(reliability_data=bias_data)))
print("Krippendorff's alpha (fluency): {:.3f}".format(krippendorff.alpha(reliability_data=fluency_data)))
print("Krippendorff's alpha (meaning): {:.3f}".format(krippendorff.alpha(reliability_data=meaning_data)))
