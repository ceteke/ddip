from tensorflow.keras import layers as tfkl


def dd_layer(inp, k):
    conv = tfkl.Conv2D(k, 1, activation='linear')(inp)
    ups = tfkl.UpSampling2D(interpolation='bilinear')(conv)
    act = tfkl.ReLU()(ups)
    b = tfkl.BatchNormalization(axis=3, center=True, scale=True)(act) # Same as Layer normalization since batch_size = 1

    return b
