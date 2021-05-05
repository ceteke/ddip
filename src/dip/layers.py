from tensorflow.keras import layers as tfkl
import tensorflow_addons as tfa


def dd_layer(inp, k):
    conv = tfkl.Conv2D(k, 1, activation='linear', use_bias=False)(inp)
    ups = tfkl.UpSampling2D(interpolation='bilinear')(conv)
    act = tfkl.ReLU()(ups)
    b = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)(act)
    # b = tfkl.BatchNormalization()(act)

    return b
