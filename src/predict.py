import tensorflow as tf
import librosa
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from genre_models.genre_model import GenreModel
from util import preprocess_wav

tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="コンフィグファイルのファイルパス")
parser.add_argument("-c_p", "--checkpoint_path", type=str, required=True, help="モデルの重みのパス")
args = parser.parse_args()

config = OmegaConf.load(args.config)
model, _ = GenreModel.from_pretrain(
    config=config,
    model_weight_path=args.checkpoint_path,
)

file = input("楽曲：")
orig_y, original_sr = librosa.load(file, sr=None, mono=False)
y = librosa.resample(orig_y, orig_sr=original_sr, target_sr=config.sampling_rate, scale=True)
y, gain_db, (mean, std) = preprocess_wav(y, config.sampling_rate)

if len(y.shape) == 1:
    y = np.array([y, y])

y = y.transpose(1, 0)
y = y[np.newaxis, ...]

predicted = model(y).numpy()
print(np.argmax(predicted), np.max(predicted))

with open("./genres.json", "r", encoding="utf-8") as f:
    genres = json.load(f)

# 上位10のインデックスを取得
top_indices = np.argsort(predicted[0])[::-1][:10]

# インデックスをジャンル名に変換
top_genres = [genres[str(i)] for i in top_indices]
top_values = predicted[0][top_indices]

# プロット
plt.figure(figsize=(10, 6))
plt.barh(top_genres[::-1], top_values[::-1])
plt.xlabel("Prediction Score")
plt.ylabel("Genre")
plt.title("Top 10 Predicted Genres")
plt.show()
