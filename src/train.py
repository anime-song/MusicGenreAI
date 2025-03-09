import os
import argparse

# 不要なログ対策
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from omegaconf import OmegaConf
from gradient_accumulator import GradientAccumulateModel

from genre_models.dataset import DataGeneratorBatch, load_from_npz
from util import allocate_gpu_memory
from genre_models.genre_model import GenreModel


def get_dataset(
    batch_size,
    patch_len,
    cache_size,
    epoch_max_steps,
    sampling_rate,
    dataset_folder_path,
):
    """
    Load and prepare the dataset for training and testing.

    Args:
        batch_size (int): The size of the batches.
        patch_len (int): The length of the patches.
        cache_size (int): The size of the cache.
        epoch_max_steps (int): The maximum number of steps per epoch.
        sampling_rate (int): The sampling rate of the audio data.

    Returns:
        tuple: A tuple containing the training, testing, and plotting data generators.
    """
    x_train, x_test, dataset = load_from_npz(directory=dataset_folder_path)

    train_gen = DataGeneratorBatch(
        files=x_train,
        dataset=dataset,
        sampling_rate=sampling_rate,
        patch_length=patch_len,
        initial_epoch=0,
        max_queue=2,
        cache_size=cache_size,
        batch_size=batch_size,
        epoch_max_steps=epoch_max_steps,
    )

    test_gen = DataGeneratorBatch(
        files=x_test,
        dataset=dataset,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        patch_length=patch_len,
        cache_size=cache_size,
        eval_mode=True,
    )

    return train_gen, test_gen


def train(
    config,
    model_weight_path=None,
):
    # モデル構築
    model, _ = GenreModel.from_pretrain(
        config=config,
        model_weight_path=model_weight_path,
    )
    model.summary()
    model = GradientAccumulateModel(
        accum_steps=config.accum_steps, inputs=model.input, outputs=model.output, mixed_precision=True
    )

    # Callback
    monitor = "val_loss"
    ckpt_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/genre_model-epoch_{epoch}_step_{batch}/generator.ckpt",
        monitor=monitor,
        verbose=1,
        save_weights_only=True,
        save_freq=config.model_save_freq,
    )

    # Optimzier
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        5e-4, decay_steps=1, decay_rate=0.999999, staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 通常の訓練
    train_gen, test_gen = get_dataset(
        config.batch_size,
        config.patch_len,
        config.cache_size,
        config.epoch_max_steps,
        config.sampling_rate,
        config.dataset_folder_path,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs",
        update_freq=10,
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.fit(
        train_gen, validation_data=test_gen, epochs=config.epochs, callbacks=[ckpt_callback_best, tensorboard_callback]
    )


if __name__ == "__main__":
    allocate_gpu_memory()

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)

    os.makedirs("./model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="コンフィグファイルのファイルパス",
    )
    parser.add_argument(
        "-c_p",
        "--checkpoint_path",
        type=str,
        default=None,
        help="モデルの重みのパス",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    train(config=config, model_weight_path=args.checkpoint_path)
