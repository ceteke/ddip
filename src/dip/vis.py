import numpy as np
import matplotlib.pyplot as plt

from .utils import psnr


def plot_training(histories, log, gt_image, noisy_image, plot_psnr=False, names=None, start=3):
    n_hist = len(histories)
    n_col = 2 if plot_psnr else 1

    fig, axs = plt.subplots(n_hist, n_col, figsize=(15, 5*n_hist), squeeze=False)

    for i, (epoch_losses, imgs_pred) in enumerate(histories):
        iter_nums = np.arange(start, len(epoch_losses)) * log

        axs[i, 0].plot(iter_nums, epoch_losses[start:])
        axs[i, 0].set_title("MSE in interations")
        axs[i, 0].grid()

        if plot_psnr:
            gt_img = np.expand_dims(gt_image, 0)

            epoch_psnr = psnr(gt_img, imgs_pred, axes=(1, 2, 3))
            inp_psnr = psnr(gt_image, noisy_image)
            fin_psnr = psnr(gt_image, imgs_pred[-1])

            axs[i, 1].plot(iter_nums, epoch_psnr[start:])
            axs[i, 1].axhline(inp_psnr, c='C3')
            axs[i, 1].set_title(f"pSNR (dB) in iterations [Final {round(fin_psnr, 2)} dB]")
            axs[i, 1].grid()

        if names is not None:
            axs[i, 0].set_ylabel(names[i])

    return fig, axs