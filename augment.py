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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="augment", description="""Script to augment dataset"""
    )
    parser.add_argument(
        "--ir",
        "-i",
        help="Directory of IR files",
        type=str,
        default="/star-data/rui/LibriheavyCSS/FAST-RIR/code_new/Generated_RIRs",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        required=True,
        help="Manifest path",
        type=str,
    )
    parser.add_argument(
        "--manifest-out",
        "-mo",
        help="Manifest output path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out",
        "-o",
        help="Output folder path",
        type=str,
        default="/star-data/rui/libriheavy_reverb",
    )
    parser.add_argument(
        "--nthreads",
        "-n",
        type=int,
        default=24,
        help="Number of threads to use",
    )

    args = parser.parse_args()
    ir_folder = args.ir
    manifest = args.manifest
    output_folder = args.out
    nthreads = args.nthreads

    irlist = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(ir_folder)
        for name in files
        if name.endswith(".wav")
    ]

    cutset = load_manifest_lazy(manifest)
    new_cuts = []
    pbar = tqdm(total=len(cutset))

    def update(*a):
        pbar.update()

    speech_folder = "/star-kw/data/libri-light/"
    output_folder = "/star-data/rui/libriheavy_reverb/"

    # pool = Pool(processes=nthreads)
    for cut in cutset:
        mono_cut = cut.to_mono()
        mono_cut.save_audio(
            output_folder + f"/{cut.id}/" + "original.flac", encoding="flac"
        )
        exit()
        ir_samples = random.choices(irlist, k=len(cut.tracks))
        tracks = cut.tracks
        new_cut = deepcopy(cut)
        try:
            for index, c in enumerate(tracks):
                if c.type == "MonoCut":
                    try:
                        speech_path = c.cut.recording.sources[0].source
                    except Exception as e:
                        print(cut.id)
                    filepath, filename = os.path.split(speech_path)
                    output_path = (
                        filepath.replace(speech_folder, output_folder) + f"/{cut.id}/"
                    )
                    if not os.path.exists(os.path.split(output_path)[0]):
                        os.makedirs(os.path.split(output_path)[0])
                    augment_data(speech_path, output_path + filename, ir_samples[index])

                    new_cut.tracks[index].cut.recording.sources[0].source = (
                        output_path + filename
                    )
                    # pool.apply_async(
                    # augment_data,
                    # args=(speech_path, output_path + filename, ir_sample),
                    # callback=update,
                    # )
                elif c.type == "MixedCut":
                    t_tracks = c.tracks
                    for i_index, c_c in enumerate(t_tracks):
                        if c_c.type == "MonoCut":
                            try:
                                speech_path = c_c.cut.recording.sources[0].source
                            except Exception as e:
                                print(cut.id)
                            filepath, filename = os.path.split(speech_path)
                            output_path = (
                                filepath.replace(speech_folder, output_folder)
                                + f"/{cut.id}/"
                            )
                            if not os.path.exists(os.path.split(output_path)[0]):
                                os.makedirs(os.path.split(output_path)[0])
                            augment_data(
                                speech_path, output_path + filename, ir_samples[index]
                            )

                            new_cut.tracks[index].cut.tracks[
                                i_index
                            ].cut.recording.sources[0].source = (output_path + filename)
                            # pool.apply_async(
                            # augment_data,
                            # args=(speech_path, output_path + filename, ir_sample),
                            # callback=update,
                            # )
                        elif c_c.type == "MixedCut":
                            t_t_tracks = c_c.tracks
                            for ii_index, c_c_c in enumerate(t_t_tracks):
                                if c_c_c.type == "MonoCut":
                                    try:
                                        speech_path = c_c_c.cut.recording.sources[
                                            0
                                        ].source
                                    except Exception as e:
                                        print(cut.id)
                                    filepath, filename = os.path.split(speech_path)
                                    output_path = (
                                        filepath.replace(speech_folder, output_folder)
                                        + f"/{cut.id}/"
                                    )
                                    if not os.path.exists(
                                        os.path.split(output_path)[0]
                                    ):
                                        os.makedirs(os.path.split(output_path)[0])
                                    augment_data(
                                        speech_path,
                                        output_path + filename,
                                        ir_samples[index],
                                    )

                                    new_cut.tracks[index].cut.tracks[
                                        i_index
                                    ].cut.tracks[ii_index].cut.recording.sources[
                                        0
                                    ].source = (
                                        output_path + filename
                                    )
                                    # pool.apply_async(
                                    # augment_data,
                                    # args=(speech_path, output_path + filename, ir_sample),
                                    # callback=update,
                                    # )
                                elif c_c_c.type == "MixedCut":
                                    t_t_t_tracks = c_c_c.tracks
                                    for iii_index, c_c_c_c in enumerate(t_t_t_tracks):
                                        if c_c_c_c.type == "MonoCut":
                                            try:
                                                speech_path = (
                                                    c_c_c_c.cut.recording.sources[
                                                        0
                                                    ].source
                                                )
                                            except Exception as e:
                                                print(cut.id)
                                            filepath, filename = os.path.split(
                                                speech_path
                                            )
                                            output_path = (
                                                filepath.replace(
                                                    speech_folder, output_folder
                                                )
                                                + f"/{cut.id}/"
                                            )
                                            if not os.path.exists(
                                                os.path.split(output_path)[0]
                                            ):
                                                os.makedirs(
                                                    os.path.split(output_path)[0]
                                                )
                                            augment_data(
                                                speech_path,
                                                output_path + filename,
                                                ir_samples[index],
                                            )

                                            new_cut.tracks[index].cut.tracks[
                                                i_index
                                            ].cut.tracks[ii_index].cut.tracks[
                                                iii_index
                                            ].cut.recording.sources[
                                                0
                                            ].source = (
                                                output_path + filename
                                            )
                                            # pool.apply_async(
                                            # augment_data,
                                            # args=(speech_path, output_path + filename, ir_sample),
                                            # callback=update,
                                            # )

        except Exception as e:
            print(cut.id)
            print(str(e))
            # pool.close()
        new_cuts.append(new_cut)
        update()

    # pool.close()
    # pool.join()
    pbar.close()
    print("Saving manifest...")
    new_cutset = CutSet.from_cuts(new_cuts)
    new_cutset.to_json(args.manifest_out)
