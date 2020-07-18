import matplotlib.pyplot as plt
import numpy as np


def display_image(im_data, axis=None):

    dpi = plt.rcParams['figure.dpi']
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    if not axis:
        ax.axis('off')

    ax.imshow(im_data)
    # ax.imshow(im_data, cmap='gray')

    plt.show()


def dim_fix(source_ini, mask_ini, target_ini, dim_ini):

    source = source_ini.copy()
    mask = mask_ini.copy()
    target = target_ini.copy()
    dim = dim_ini.copy()

    # get image shape and offset
    Hs, Ws, _ = source.shape
    Ht, Wt, _ = target.shape
    Ho, Wo = dim

    # adjust source and mask if offset is negative.
    # if mask is rolled eg. from the top it rolls
    # to the bottom, crop the rolled portion
    if(Ho < 0):
        mask = np.roll(mask, Ho, axis=0)
        source = np.roll(source, Ho, axis=0)
        mask[Hs+Ho:, :, :] = 0  # added because Ho < 0
        source[Hs+Ho:, :, :] = 0
        Ho = 0
    if(Wo < 0):
        mask = np.roll(mask, Wo, axis=1)
        source = np.roll(source, Wo, axis=1)
        mask[:, Ws+Wo:, :] = 0
        source[:, Ws+Wo:, :] = 0
        Wo = 0

    # mask region on target
    H_min = Ho
    H_max = min(Ho + Hs, Ht)
    W_min = Wo
    W_max = min(Wo + Ws, Wt)

    # crop source and mask if they lie outside the bounds of the target
    source = source[0:min(Hs, Ht-Ho), 0:min(Ws, Wt-Wo), :]
    mask = mask[0:min(Hs, Ht-Ho), 0:min(Ws, Wt-Wo), :]

    return source,  mask,  target, [H_min, H_max, W_min, W_max]
