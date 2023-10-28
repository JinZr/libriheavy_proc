import argparse
import os
import random
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from lhotse import CutSet, load_manifest_lazy
from tqdm import tqdm

import utility

parser = argparse.ArgumentParser()
parser.add_argument(
    "--manifest",
    type=str,
    required=True,
    help="Path to the manifest JSON",
)
parser.add_argument(
    "--ir",
    "-i",
    help="Directory of IR files",
    type=str,
    default="/star-data/rui/LibriheavyCSS/FAST-RIR/code_new/Generated_RIRs",
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
ir_folder = args.ir


def augment_data(speech_path, output_path, irfile_path):
    speech, fs_s = sf.read(speech_path)
    if len(speech.shape) != 1:
        speech = speech[:, 0]
    if np.issubdtype(speech.dtype, np.integer):
        speech = utility.pcm2float(speech, "float32")
    # convolution
    if irfile_path:
        IR, fs_i = sf.read(irfile_path)
        if len(IR.shape) != 1:
            IR = IR[:, 0]
        if np.issubdtype(IR.dtype, np.integer):
            IR = utility.pcm2float(IR, "float32")
        speech = utility.convert_samplerate(speech, fs_s, fs_i)
        fs_s = fs_i
        # eliminate delays due to direct path propagation
        direct_idx = np.argmax(np.fabs(IR))
        # print('speech {} direct index is {} of total {} samples'.format(speech_path, direct_idx, len(IR)))
        temp = utility.smart_convolve(speech, IR[direct_idx:])
        speech = np.array(temp)
    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio
    sf.write(output_path, speech, fs_s)


# Load the manifest and convert it to a CutSet
cutset = load_manifest_lazy(manifest)
reverb_cuts = []

audio_path = f"/star-data/rui/libriheavy_reverb/{dset}"
irlist = [
    os.path.join(root, name)
    for root, dirs, files in os.walk(ir_folder)
    for name in files
    if name.endswith(".wav")
]

if not os.path.exists(audio_path):
    os.makedirs(audio_path)

for cut in tqdm(cutset, desc="Processing cuts"):
    out_cut = deepcopy(cut)
    flac_f = out_cut.recording.sources[0].source
    ir_sample = random.choice(irlist)
    augment_data(
        flac_f,
        audio_path + flac_f.split("/")[-1],
        ir_sample,
    )
    out_cut.recording.sources[0].source = audio_path + flac_f.split("/")[-1]

    reverb_cuts.append(out_cut)

reverb_cutset = CutSet.from_cuts(reverb_cuts)
reverb_cutset.to_jsonl(output + "/" + manifest.split(".")[0] + "_rir.jsonl.gz")


reverb_cutset.describe(full=True)
