import time
import numpy as np


def train(model, niter, img, gt_img, info=100, log=1, z=None, tqdm=None):
    img_t = np.expand_dims(img, 0)

    pred_size = [niter // log] + list(img.shape)
    pred_imgs = np.zeros(pred_size)

    losses = np.zeros(niter // log)

    if z is None:
        z = np.random.randn(*model.input_shape)

    if tqdm:
        progress_bar = tqdm.trange(niter, desc='Training', leave=True)
    else:
        progress_bar = None

    for i in range(niter):
        loss = model.fit(z, img_t, batch_size=1, epochs=1, verbose=0).history['loss'][0]

        if i % info == 0:
            if progress_bar:
                progress_bar.set_description(f"Iteration {i} loss: {round(loss, 4)}")
                progress_bar.update(info)
                progress_bar.refresh()
            else:
                print(f"{i}:", loss)

        if i % log == 0:
            pred_img = model.predict(z)
            losses[i // log] = np.mean(np.square(pred_img - gt_img))
            pred_imgs[i // log] = pred_img

    if progress_bar is not None:
        progress_bar.close()

    return losses, pred_imgs
