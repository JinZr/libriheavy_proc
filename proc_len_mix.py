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
new_cutset = cuts.filter(lambda x: x.duration < 30 or len(x.tracks) == 2)

new_cutset.describe(full=True)
new_cutset.to_jsonl(args.output)
