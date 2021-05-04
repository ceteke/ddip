import tensorflow as tf
import tensorflow.keras.layers as tkl

from .layers import dd_layer


def get_loss(mask):
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, 0)

    def _loss(y_gold, y_pred):
        return tf.reduce_mean(tf.square(y_gold - y_pred) * mask)

    return _loss


def get_dip_model(input_size, mask, optimizer=None):
    model = hour_glass(input_size)
    loss = get_loss(mask)
    optimier = tf.optimizers.Adam() if optimizer is None else optimizer

    model.compile(optimizer=optimier, loss=loss)
    return model


def get_dd_model(input_size, mask, optimizer=None):
    model = decoder(input_size)
    loss = get_loss(mask)
    optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

    model.compile(optimizer=optimizer, loss=loss)
    return model


def decoder(input_size, k=64):
    inputs = tkl.Input(input_size)
    l1 = dd_layer(inputs, k)
    l2 = dd_layer(l1, k)
    l3 = dd_layer(l2, k)
    l4 = dd_layer(l3, k)
    l5 = dd_layer(l4, k)
    out = tkl.Conv2D(3, 1, activation='sigmoid')(l5)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def hour_glass(input_size, sigm=True):
    act = 'sigmoid' if sigm else 'linear'

    inputs = tkl.Input(input_size)
    conv1 = tkl.Conv2D(32, 3, activation='relu', padding='same', strides=2)(inputs)
    conv2 = tkl.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
    conv3 = tkl.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv2)
    conv4 = tkl.Conv2D(256, 3, activation='relu', padding='same', strides=2)(conv3)

    up5 = tkl.Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2)(conv4)
    up6 = tkl.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2)(up5)
    up7 = tkl.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(up6)
    up8 = tkl.Conv2DTranspose(3, 3, activation=act, padding='same', strides=2)(up7)

    model = tf.keras.Model(inputs=inputs, outputs=up8)

    return model
