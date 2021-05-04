import numpy as np


def train(model, niter, img, info=100, log=1, z=None):
    img_t = np.expand_dims(img, 0)

    pred_size = [niter // log] + list(img.shape)
    pred_imgs = np.zeros(pred_size)

    losses = np.zeros(niter // log)

    for i in range(niter):
        loss = model.fit(z, img_t, batch_size=1, epochs=1, verbose=0).history['loss'][0]

        if i % info == 0:
            print(f"{i}:", loss)

        if i % log == 0:
            losses[i // log] = loss
            pred_imgs[i // log] = model.predict(z)

    return losses, pred_imgs
