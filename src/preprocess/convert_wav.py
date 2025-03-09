import os
import json
import argparse
import librosa
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pyloudnorm as pyln
import h5py
import resampy

lock = multiprocessing.Lock()


def save_to_h5(filepath, dataset_name, y, genre_id):
    with lock:
        with h5py.File(filepath, mode="a") as h5:
            group = h5.require_group("music")
            if dataset_name not in group:
                dset = group.create_dataset(name=dataset_name, data=y, dtype=y.dtype, shape=y.shape)
                # 属性として genre_id を保存
                dset.attrs["genre_id"] = genre_id


def exists_dataset(dataset_name, dataset_path):
    filepath = os.path.join(dataset_path, "dataset") + ".hdf5"
    if not os.path.exists(filepath):
        return False
    with lock:
        with h5py.File(filepath, mode="r") as h5:
            group_path = "/music"
            if group_path in h5 and f"{group_path}/{dataset_name}" in h5:
                return True
            else:
                return False


def process_file(entry, audio_file_path, sampling_rate, dataset_path):
    spotify_id = entry["spotify_id"]
    genre_id = entry["genre_id"]
    # 保存時のデータセット名は "spotify_id_genre_id"
    dataset_name = f"{spotify_id}_{genre_id}"

    # 指定パスからファイルを構成: audio_file_path + spotify_id.mp3
    filepath_audio = os.path.join(audio_file_path, f"{spotify_id}.mp3")
    if not os.path.exists(filepath_audio):
        print(f"File {filepath_audio} not found, skipping.")
        return

    try:
        if exists_dataset(dataset_name, dataset_path):
            # print(f"{dataset_name} already exists, skipping.")
            return

        print("Processing:", dataset_name)
        # 音声読み込み
        y, sr = librosa.load(filepath_audio, sr=None, mono=False)
        y = resampy.resample(y, sr_orig=sr, sr_new=sampling_rate, filter="sinc_window", num_zeros=32)
        if len(y.shape) == 1:
            y = np.array([y, y])

        # ラウドネス正規化
        meter = pyln.Meter(sampling_rate)
        loudness = meter.integrated_loudness(y.T)
        y_norm = pyln.normalize.loudness(y.T, loudness, -24.0)
        y_norm = y_norm.T.astype("float16")

        # HDF5 に保存
        h5_filepath = os.path.join(dataset_path, "dataset") + ".hdf5"
        save_to_h5(h5_filepath, dataset_name, y_norm, genre_id)
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file_path", default="./songs")
    parser.add_argument("--sampling_rate", default=44100)
    parser.add_argument("--dataset_path", default="./Dataset/Processed")
    parser.add_argument("--mapping_json", default="songs.json")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    n_proc = 4
    Parallel(n_jobs=n_proc, backend="multiprocessing", verbose=1)(
        [
            delayed(process_file)(entry, args.audio_file_path, args.sampling_rate, args.dataset_path)
            for entry in mapping_data
        ]
    )
