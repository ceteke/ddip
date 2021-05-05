import numpy as np
import PIL


def read_img(path):
    return np.asarray(PIL.Image.open(path), dtype='float') / 255


def thanos_noise(img, p1):
    mask = np.random.choice([0, 1], size=img.shape[:-1], p=[1 - p1, p1])
    mask = np.repeat(np.expand_dims(mask, -1), 3, -1)
    return img * mask, mask


def gauss_noise(img, std):
    return np.clip(img + np.random.randn(*img.shape) * std, 0, 1), np.ones_like(img)


def remove_center(img, size):
    w, h, _ = img.shape

    x1 = w // 2 - size // 2
    x2 = w // 2 + size // 2
    y1 = h // 2 - size // 2
    y2 = h // 2 + size // 2

    mask = np.ones_like(img)
    mask[x1:x2, y1:y2, :] = 0

    return img * mask, mask


def psnr(gt, y, axes=None):
    return -10 * np.log10(np.mean(np.square(gt - y), axis=axes))