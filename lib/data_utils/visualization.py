# ---------------------------------------------------------------
# SNIPER: Efficient Multi-scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# by Mahyar Najibi
# ---------------------------------------------------------------
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


def visualize_dets(im, dets, scale, pixel_means, class_names, threshold=0.5, save_path='debug.png', transform=True):
    if transform:
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

def vis_polys(polys, im_path, crop, scale):
    im = misc.imread(im_path)
    im = im[:, ::-1, :]
    for obj in range(len(polys)):
        plt.imshow(im)
        n_seg = len(polys[obj])
        for j in range(n_seg):
            cur_len = len(polys[obj][j])
            for k in range(cur_len/2):
                point = plt.Circle((polys[obj][j][2*k], polys[obj][j][2*k+1]), radius=1, color='red')
                plt.gca().add_patch(point)
        num = np.random.randint(0,100000)
        plt.savefig('debug/visualization/debug_{}_{}.png'.format(num, obj))
        plt.clf()
        plt.cla()
        plt.close()
        imc = im[int(crop[1]):int(crop[3]), int(crop[0]):int(crop[2])]
        imc2 = misc.imresize(imc, scale)
        plt.imshow(imc2)
        h,w,_ = np.shape(imc2)
        n_seg = len(polys[obj])
        for j in range(n_seg):
            cur_len = len(polys[obj][j])
            for k in range(cur_len/2):
                x1 = (polys[obj][j][2*k]-crop[0])*scale
                y1 = scale*(polys[obj][j][2*k+1]-crop[1])
                x1 = min(max(0, x1), w)
                y1 = min(max(0, y1), h)
                point = plt.Circle((x1, y1), radius=1, color='red')
                plt.gca().add_patch(point)

        plt.savefig('debug/visualization/debug_{}_{}_c.png'.format(num, obj))
        plt.clf()
        plt.cla()
        plt.close()
