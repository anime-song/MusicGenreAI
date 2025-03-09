import tensorflow as tf

from losses import MelSpectrogram
from model_utils import (
    RMSNorm,
    PositionalEncoding,
)
from latent_transformer import LatentTransformer


class DualPathBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_freq_transformers=1,
        num_time_transformers=1,
        use_mix_ln=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_time_transformers = num_time_transformers
        self.num_freq_transformers = num_freq_transformers

        if num_time_transformers > 0:
            self.time_pos_enc = PositionalEncoding(hidden_size, max_shift_size=40, dtype=tf.float32)
            self.time_transformer = LatentTransformer(
                hidden_size, num_layers=num_time_transformers, mix_layer_norm=use_mix_ln
            )
            self.time_linear = tf.keras.layers.Dense(hidden_size)
            self.time_norm = RMSNorm()

        if num_freq_transformers > 0:
            self.freq_pos_enc = PositionalEncoding(hidden_size, max_shift_size=40, dtype=tf.float32)
            self.freq_transformer = LatentTransformer(
                hidden_size, num_layers=num_freq_transformers, mix_layer_norm=use_mix_ln
            )
            self.freq_linear = tf.keras.layers.Dense(hidden_size)
            self.freq_norm = RMSNorm()

    def call(self, inputs, training=False):
        # inputs: (batch , time, freq, num_channels)
        batch_size = tf.shape(inputs)[0]
        time = tf.shape(inputs)[1]
        freq = tf.shape(inputs)[2]
        num_channels = self.hidden_size

        # freq
        if self.num_freq_transformers > 0:
            residual = inputs
            inputs = tf.reshape(inputs, [batch_size * time, freq, num_channels])
            inputs = self.freq_pos_enc(inputs, training=training)
            inputs = self.freq_transformer(inputs, training=training)
            inputs = self.freq_linear(inputs)
            inputs = self.freq_norm(inputs)
            inputs = tf.reshape(inputs, [batch_size, time, freq, num_channels])
            inputs = inputs + residual

        # time
        if self.num_time_transformers > 0:
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            residual = inputs
            inputs = tf.reshape(inputs, [batch_size * freq, time, num_channels])

            inputs = self.time_pos_enc(inputs, training=training)
            inputs = self.time_transformer(inputs, training=training)
            inputs = self.time_linear(inputs)
            inputs = self.time_norm(inputs)

            inputs = tf.reshape(inputs, [batch_size, freq, time, num_channels])
            inputs = inputs + residual
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
        return inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_sizes=[32, 64, 128, 256],
        dual_path_blocks=[4, 4, 4, 4],
        down_sampling_factors=[[2, 2], [2, 2], [2, 2], [2, 2]],
        num_freq_transformers=1,
        num_time_transformers=1,
        use_mix_ln=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.blocks = []
        for i in range(len(hidden_sizes)):
            self.blocks.append(
                tf.keras.layers.Conv2D(hidden_sizes[i], kernel_size=(1, 1), strides=(1, 1), padding="same")
            )
            for _ in range(dual_path_blocks[i]):
                self.blocks.append(
                    DualPathBlock(
                        hidden_size=hidden_sizes[i],
                        num_freq_transformers=num_freq_transformers,
                        num_time_transformers=num_time_transformers,
                        use_mix_ln=use_mix_ln,
                    )
                )

            self.blocks.append(
                tf.keras.layers.Conv2D(
                    hidden_sizes[i],
                    kernel_size=(down_sampling_factors[i][0] * 2, down_sampling_factors[i][1] * 2),
                    strides=down_sampling_factors[i],
                    padding="same",
                )
            )

    def call(self, inputs, training=False):
        for layer in self.blocks:
            inputs = layer(inputs, training=training)
        return inputs


class GenreEncoderModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.encoder = Encoder(
            hidden_sizes=config.hidden_sizes,
            dual_path_blocks=config.dual_path_blocks,
            down_sampling_factors=config.down_sampling_factors,
            num_freq_transformers=config.num_freq_transformer,
            num_time_transformers=config.num_time_transformer,
            use_mix_ln=config.use_mix_ln,
        )
        self.num_channels = config.num_channels

        self.mel_layer = MelSpectrogram(
            window_length=config.window_length,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            sampling_rate=config.sampling_rate,
            fmax=None,
            n_mels=config.n_mels,
            dtype=tf.float32,
        )

    @classmethod
    def from_pretrain(
        cls,
        config,
        model_weight_path=None,
    ):
        model_input = tf.keras.layers.Input(shape=(None, config.num_channels))

        intermediate_model = cls(config=config)
        outputs = intermediate_model(model_input)
        model = tf.keras.Model(inputs=[model_input], outputs=outputs)

        if model_weight_path is not None:
            model.load_weights(model_weight_path)

        return model, intermediate_model

    def preprocess(self, signals):
        mel = self.mel_layer(signals)
        return mel

    def call(self, inputs, training=False):
        inputs = self.preprocess(inputs)
        # inputs: (B, T, Fr, C)
        mean = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(inputs, axis=[1, 2, 3], keepdims=True)
        inputs = (inputs - mean) / (std + 1e-5)
        inputs = self.encoder(inputs, training=training)
        return inputs


class GenreModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_separate = 2

        self.encoder = GenreEncoderModel(config)
        self.gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")
        self.genre = tf.keras.layers.Dense(config.num_genres, activation="softmax", dtype=tf.float32, name="genre")

    @classmethod
    def from_pretrain(
        cls,
        config,
        model_weight_path=None,
    ):
        model_input = tf.keras.layers.Input(shape=(None, config.num_channels))

        intermediate_model = cls(config=config)
        outputs = intermediate_model(model_input)
        model = tf.keras.Model(inputs=[model_input], outputs=outputs)

        if model_weight_path is not None:
            model.load_weights(model_weight_path)

        return model, intermediate_model

    def call(self, inputs, training=False):
        inputs = self.encoder(inputs, training=training)

        genre = self.gap(inputs)
        genre = self.genre(genre)
        return genre
