import argparse
import os
import random

from lhotse import load_manifest_lazy
from lhotse.cut import CutSet
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--manifest",
    type=str,
    required=True,
    help="Path to the manifest JSON",
)
parser.add_argument(
    "--dset",
    type=str,
    required=True,
    help="Path to the manifest JSON",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the output file",
)
args = parser.parse_args()

manifest = args.manifest
dset = args.dset
output = args.output

# Load the manifest and convert it to a CutSet
cutset = load_manifest_lazy(manifest)
new_cutset = []
new_mono_cutset = []

audio_path = f"/star-data/rui/libriheavy_ovlp/{dset}"

if not os.path.exists(audio_path):
    os.makedirs(audio_path)

for cut in tqdm(cutset, desc="Processing cuts"):
    try:
        out_cut = cut.drop_features()
        snrs = [random.uniform(-5, 5) for _ in range(len(cut.tracks))]
        for i, (track, snr) in enumerate(zip(out_cut.tracks, snrs)):
            if i == 0:
                # Skip the first track since it is the reference
                continue
            track.snr = snr
        new_cutset.append(out_cut)

        mono_cut = cut.to_mono()
        new_mono_cutset.append(mono_cut)

        mono_cut.save_audio(audio_path + "/" + mono_cut.id + ".flac")
    except:
        continue

new_cutset = CutSet.from_cuts(new_cutset)
new_cutset.to_jsonl(output + "/" + manifest.split(".")[0] + "_snr_aug.jsonl.gz")

mono_cutset = CutSet.from_cuts(new_mono_cutset)
mono_cutset.to_jsonl(output + "/" + manifest.split(".")[0] + "_snr_aug_mono.jsonl.gz")

mono_cutset.describe(full=True)
