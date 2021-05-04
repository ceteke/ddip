import numpy as np
import matplotlib.pyplot as plt

from .utils import psnr


def plot_training(histories, log, gt_image, noisy_image):
    n_hist = len(histories)
    fig, axs = plt.subplots(n_hist, 2, figsize=(15, 5*n_hist), squeeze=False)

    for i, (epoch_losses, imgs_pred) in enumerate(histories):
        # PSNR computation
        gt_img = np.expand_dims(gt_image, 0)
        epoch_psnr = psnr(gt_img, imgs_pred, axes=(1, 2, 3))
        inp_psnr = psnr(gt_image, noisy_image)

        iter_nums = np.arange(len(epoch_psnr)) * log

        axs[i, 0].plot(iter_nums, epoch_losses)
        axs[i, 0].set_title("MSE in interations")
        axs[i, 0].grid()
        axs[i, 1].plot(iter_nums, epoch_psnr)
        axs[i, 1].axhline(inp_psnr, c='C3')
        axs[i, 1].set_title(f"pSNR (dB) in iterations [{round(inp_psnr, 2)} dB]")
        axs[i, 1].grid()

    return fig, axs