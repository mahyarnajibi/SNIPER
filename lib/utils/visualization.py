import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np


def transform_im(im, pixel_means):
    im = im.copy()
    # put channel back
    im = im.transpose((1, 2, 0))
    im += pixel_means
    return im.astype(np.uint8)


def visualize_dets(im, dets, scale, pixel_means, class_names, threshold=0.5, save_path='debug.png'):
    im = transform_im(im, np.array(pixel_means)[[2, 1, 0]])
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__': continue
        color = (random.random(), random.random(), random.random())
        for det in dets[j]:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()
