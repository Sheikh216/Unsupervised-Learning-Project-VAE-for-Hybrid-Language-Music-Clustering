"""
Variational Autoencoder (VAE) implementations in TensorFlow/Keras.
Includes:
- MLP-VAE (basic)
- Beta-VAE
- Conditional VAE (CVAE)
- Convolutional VAE (for spectrograms)
- Simple Autoencoder (baseline)
"""

from typing import Tuple, Optional, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


class Sampling(layers.Layer):
    """Reparameterization trick layer: z = mu + exp(0.5 * log_var) * eps"""
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(0.5 * log_var) * epsilon


def build_mlp_encoder(input_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
    inputs = layers.Input(shape=(input_dim,), name="encoder_input")
    x = inputs
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
    mu = layers.Dense(latent_dim, name="z_mean")(x)
    log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([mu, log_var])
    return models.Model(inputs, [mu, log_var, z], name="encoder")


def build_mlp_decoder(output_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")
    x = latent_inputs
    for h in hidden_dims[::-1]:
        x = layers.Dense(h, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="decoder_output")(x)
    return models.Model(latent_inputs, outputs, name="decoder")


class VAE(tf.keras.Model):
    """Basic MLP VAE. Supports beta parameter (Beta-VAE)."""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: Tuple[int, ...] = (256, 128), beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = build_mlp_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = build_mlp_decoder(input_dim, hidden_dims, latent_dim)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            # MSE reconstruction loss
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        _, _, z = self.encoder(inputs, training=False)
        return self.decoder(z, training=False)

    def encode(self, x: np.ndarray) -> np.ndarray:
        mu, _, _ = self.encoder.predict(x, verbose=0)
        return mu

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x, verbose=0)


class CVAE(tf.keras.Model):
    """Conditional VAE with one-hot condition vector c concatenated to inputs and z."""
    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int = 16, hidden_dims: Tuple[int, ...] = (256, 128), beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder with condition
        x_in = layers.Input(shape=(input_dim,))
        c_in = layers.Input(shape=(cond_dim,))
        x = layers.Concatenate()([x_in, c_in])
        for h in hidden_dims:
            x = layers.Dense(h, activation="relu")(x)
        mu = layers.Dense(latent_dim)(x)
        log_var = layers.Dense(latent_dim)(x)
        z = Sampling()([mu, log_var])
        self.encoder = models.Model([x_in, c_in], [mu, log_var, z], name="cvae_encoder")

        # Decoder with condition
        z_in = layers.Input(shape=(latent_dim,))
        dc_in = layers.Input(shape=(cond_dim,))
        d = layers.Concatenate()([z_in, dc_in])
        for h in hidden_dims[::-1]:
            d = layers.Dense(h, activation="relu")(d)
        out = layers.Dense(input_dim, activation="linear")(d)
        self.decoder = models.Model([z_in, dc_in], out, name="cvae_decoder")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # Support both passing only inputs (x, c) and inputs with targets ((x, c), y)
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], (tuple, list)):
            (x, c), _ = data
        else:
            (x, c) = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder([x, c], training=True)
            reconstruction = self.decoder([z, c], training=True)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        x, c = inputs
        mu, _, z = self.encoder([x, c], training=False)
        return self.decoder([z, c], training=False)

    def encode(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        mu, _, _ = self.encoder.predict([x, c], verbose=0)
        return mu

    def reconstruct(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        mu, _, z = self.encoder.predict([x, c], verbose=0)
        return self.decoder.predict([z, c], verbose=0)


class ConvVAE(tf.keras.Model):
    """Convolutional VAE for spectrogram inputs shaped (H, W, 1)."""
    def __init__(self, input_shape=(128, 128, 1), latent_dim: int = 32, beta: float = 1.0):
        super().__init__()
        self.input_shape_spec = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        encoder_inputs = layers.Input(shape=input_shape)
        x = encoder_inputs
        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        mu = layers.Dense(latent_dim, name="z_mean")(x)
        log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([mu, log_var])
        self.encoder = models.Model(encoder_inputs, [mu, log_var, z], name="conv_encoder")

        # Decoder
        decoder_inputs = layers.Input(shape=(latent_dim,))
        # compute the spatial dims after three downsamplings (divide by 8)
        h, w = input_shape[0] // 8, input_shape[1] // 8
        x = layers.Dense(h * w * 128, activation="relu")(decoder_inputs)
        x = layers.Reshape((h, w, 128))(x)
        x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        outputs = layers.Conv2D(1, 3, padding="same", activation="linear")(x)
        self.decoder = models.Model(decoder_inputs, outputs, name="conv_decoder")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=[1, 2, 3]))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def encode(self, x: np.ndarray) -> np.ndarray:
        mu, _, _ = self.encoder.predict(x, verbose=0)
        return mu

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x, verbose=0)


class Autoencoder(tf.keras.Model):
    """Simple fully connected autoencoder as a baseline."""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: Tuple[int, ...] = (256, 128)):
        super().__init__()
        self.encoder_net = build_mlp_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder_net = build_mlp_decoder(input_dim, hidden_dims, latent_dim)

    def call(self, inputs):
        _, _, z = self.encoder_net(inputs, training=False)
        return self.decoder_net(z, training=False)

    def encode(self, x: np.ndarray) -> np.ndarray:
        mu, _, _ = self.encoder_net.predict(x, verbose=0)
        return mu

    def fit_ae(self, x: np.ndarray, batch_size=128, epochs=20, validation_data=None):
        # Train AE by minimizing MSE between input and reconstruction
        inp = layers.Input(shape=(x.shape[1],))
        mu, _, z = self.encoder_net(inp)
        out = self.decoder_net(z)
        model = models.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        return model.fit(x, x, batch_size=batch_size, epochs=epochs, validation_data=(validation_data, validation_data) if validation_data is not None else None, verbose=1)
