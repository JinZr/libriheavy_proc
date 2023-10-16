import argparse

import lhotse
from lhotse import CutSet, MonoCut, load_manifest_lazy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=str, help="Path to the manifest JSON")
parser.add_argument("--output", type=str, help="Path to the output file")
args = parser.parse_args()

# Load the manifest and convert it to a CutSet
# Note: we use the load_manifest_lazy() function to avoid loading the audio files
# into memory.
cuts = CutSet.from_manifests(load_manifest_lazy(args.manifest))
new_cuts = []
for cut in tqdm(cuts):
    # cut["text"] = cut["custom"]["texts"][1]
    assert len(cut["supervisions"]) == 1

    cut["supervisions"][0]["text"] = cut["custom"]["texts"][1]
    cut["supervisions"][0]["custom"]["orig_supervision"] = cut["supervisions"][0]

    new_cuts.append(cut)

new_cutset = CutSet.from_cuts(new_cuts)
new_cutset.describe(full=True)
new_cutset.to_file(args.output)
