import argparse
import copy

import lhotse
from lhotse import CutSet, MonoCut, load_manifest_lazy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=str, help="Path to the manifest JSON")
args = parser.parse_args()

# Load the manifest and convert it to a CutSet
# Note: we use the load_manifest_lazy() function to avoid loading the audio files
# into memory.
# cuts = CutSet.from_jsonl_lazy(args.manifest)
cuts = load_manifest_lazy(args.manifest)

cuts.describe(full=True)
