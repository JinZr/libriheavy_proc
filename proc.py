import argparse
import copy

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
# cuts = CutSet.from_jsonl_lazy(args.manifest)
cuts = load_manifest_lazy(args.manifest)
new_cuts = []
for cut in tqdm(cuts):
    # cut["text"] = cut["custom"]["texts"][1]

    d_cut = cut.to_dict()

    assert len(d_cut["supervisions"]) == 1

    d_cut["supervisions"][0]["text"] = d_cut["supervisions"][0]["custom"]["texts"][0]
    orig_supervision = copy.deepcopy(d_cut["supervisions"][0]["custom"])
    d_cut["custom"].update(orig_supervision)

    c_cut = MonoCut.from_dict(d_cut)

    new_cuts.append(c_cut)

new_cutset = CutSet.from_cuts(new_cuts)
new_cutset.describe(full=True)
new_cutset.to_jsonl(args.output)
