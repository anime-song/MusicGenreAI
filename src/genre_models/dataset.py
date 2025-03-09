import time
import random
import threading
import copy
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import h5py


def load_from_npz(directory="./Dataset/Processed"):
    h5 = h5py.File(directory + "/dataset.hdf5", mode="r")
    group = h5.require_group("music")

    # メモリーリークの可能性あり
    datasetList = [name for name in group if isinstance(group[name], h5py.Dataset)]
    labels = [group[name].attrs.get("genre_id") for name in datasetList]
    dataset = group

    train_files, test_files = train_test_split(datasetList, test_size=0.05, random_state=42, stratify=labels)

    return train_files, test_files, dataset


def load_data(self):
    while not self.is_epoch_end:
        should_added_queue = len(self.data_cache_queue) < self.max_queue
        while should_added_queue:
            self._load_cache(self.file_index)
            should_added_queue = len(self.data_cache_queue) < self.max_queue

        time.sleep(0.1)


class DataLoader:
    def __init__(
        self,
        files,
        dataset,
        seq_len,
        sampling_rate,
        max_sample_length=3600,
        max_queue=1,
        cache_size=100,
        num_threads=1,
        eval_mode=False,
    ):
        self.dataset = dataset
        self.files_index = files.index
        self.files = sorted(set(copy.deepcopy(files)), key=self.files_index)
        self.file_index = 0
        self.data_cache = {}
        self.data_cache_queue = []
        self.eval_mode = eval_mode

        self.sampling_rate = sampling_rate
        self.max_sample_length = max_sample_length
        self.cache_size = cache_size
        self.max_queue = max_queue
        self.num_threads = num_threads
        self.is_epoch_end = False
        self.seq_len = seq_len
        self.threads = []
        self.start()

    def _load_cache(self, start_idx):
        cache = {}
        self.file_index += self.cache_size
        for i in range(self.cache_size):
            idx = np.random.randint(0, len(self.files))
            if self.eval_mode:
                idx = (start_idx + i) % len(self.files)
            file_name = self.files[idx]
            data = self.dataset[file_name]

            n_frames = data.shape[1]
            n_frames = min(self.max_sample_length * self.sampling_rate, n_frames)
            if n_frames <= self.seq_len:
                start = 0
            else:
                start = np.random.randint(0, n_frames - self.seq_len)

            if self.eval_mode:
                start = 0

            spect = data[:, start : start + self.seq_len].astype("float32")
            spect = np.nan_to_num(spect, nan=0, posinf=0, neginf=0)
            spect = (spect - np.mean(spect)) / (np.std(spect) + 1e-5)

            genre_id = data.attrs.get("genre_id")
            cache[file_name] = [spect, genre_id]

        self.data_cache_queue.append(cache)

    def on_epoch_end(self):
        self.is_epoch_end = True
        self.join()

        self.is_epoch_end = False

        self.file_index = 0
        self.data_cache.clear()
        self.data_cache_queue.clear()

        self.start()

    def start(self):
        for _ in range(self.num_threads):
            thread = threading.Thread(target=load_data, args=(self,))
            thread.start()
            self.threads.append(thread)

    def join(self):
        for thread in self.threads:
            thread.join()
        self.threads = []

    def select_data(self):
        while len(self.data_cache) <= 0:
            if len(self.data_cache_queue) >= 1:
                self.data_cache = self.data_cache_queue.pop(0)
                break

            time.sleep(0.1)

        if self.eval_mode:
            file_name, data = list(self.data_cache.items())[0]
        else:
            file_name, data = random.choice(list(self.data_cache.items()))

        del self.data_cache[file_name]
        return data

    def __len__(self):
        return len(self.files)


class DataGeneratorBatch(keras.utils.Sequence):
    def __init__(
        self,
        files: list,
        dataset,
        sampling_rate,
        batch_size=32,
        patch_length=128,
        initial_epoch=0,
        max_queue=1,
        cache_size=500,
        num_threads=1,
        epoch_max_steps=None,
        eval_mode=False,
    ):
        print("files size:{}".format(len(files)))
        self.eval_mode = eval_mode

        self.dataloader = DataLoader(
            files,
            dataset,
            sampling_rate=sampling_rate,
            seq_len=patch_length,
            max_queue=max_queue,
            cache_size=cache_size,
            num_threads=num_threads,
            eval_mode=eval_mode,
        )

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.patch_length = patch_length

        if epoch_max_steps is not None:
            self.batch_len = epoch_max_steps
        else:
            total_seq_length = 0
            for file in files:
                length = dataset[file].shape[1]
                total_seq_length += (length // self.patch_length) * self.patch_length

            self.batch_len = int((total_seq_length // self.patch_length // self.batch_size)) + 1

        # データ読み込み
        self.epoch = initial_epoch

    def on_epoch_end(self):
        self.dataloader.on_epoch_end()
        self.epoch += 1

    def __getitem__(self, index):
        X = np.full((self.batch_size, self.patch_length, 2), 0, dtype="float32")
        Y = np.full((self.batch_size, 1), 0, dtype="int32")

        select_num = self.batch_size
        for batch in range(select_num):
            data = self.dataloader.select_data()
            audio_data = data[0]

            nframes = audio_data.shape[1]
            X[batch, :nframes, 0] = audio_data[0]
            X[batch, :nframes, 1] = audio_data[1]
            Y[batch, 0] = int(data[1])

        return X, Y

    def __len__(self):
        return self.batch_len
