import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


""" Build the Auto-Encoder model """
class AutoEncoder(Model):
    def __init__(self, n_input):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
                            layers.Dense(n_input, activation="relu"),
                            layers.Dense(32, activation="relu"),
                            layers.Dense(16, activation="relu")
                            ])

        self.decoder = tf.keras.Sequential([
                            layers.Dense(16, activation="relu"),
                            layers.Dense(32, activation="relu"),
                            layers.Dense(n_input, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_uncompiled_model(n_input):
    uncompiled_model = AutoEncoder(n_input)  # an object
    return uncompiled_model


def get_compiled_model(n_input):
    model = get_uncompiled_model(n_input)  # an uncompiled model
    model.compile(optimizer='adam', loss='mae')  # a compiled model
    return model