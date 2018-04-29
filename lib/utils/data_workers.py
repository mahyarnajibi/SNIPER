import cv2
import mxnet as mx
import numpy as np

class im_worker(object):
    def __init__(self, cfg, crop_size=None):
        self.cfg = cfg
        self.crop_size = crop_size

    def worker(self, data):

        imp = data[0]
        flipped = data[2]
        pixel_means = self.cfg.network.PIXEL_MEANS
        im = cv2.imread(imp, cv2.IMREAD_COLOR)
        # Flip the image
        if flipped:
            im = im[:, ::-1, :]

        # Crop if required
        if self.crop_size:
            crop = data[1]
            max_size = [self.crop_size, self.crop_size]
            im = im[int(crop[0][1]):int(crop[0][3]), int(crop[0][0]):int(crop[0][2]), :]
            scale = crop[1]
        else:
            max_size = data[1]
            # Compute scale based on config
            min_target_size = self.cfg.SCALES[0][0]
            max_target_size = self.cfg.SCALES[0][1]
            im_size_min = np.min(im.shape[0:2])
            im_size_max = np.max(im.shape[0:2])
            scale = float(min_target_size) / float(im_size_min)
            if np.round(scale * im_size_max) > max_target_size:
                scale = float(max_target_size) / float(im_size_max)
        # Resize the image
        try:
            im = cv2.resize(im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        except:
            print 'Image Resize Failed!'

        rim = np.zeros((3, max_size[0], max_size[1]), dtype=np.float32)
        d1m = min(im.shape[0], max_size[0])
        d2m = min(im.shape[1], max_size[1])
        if not self.cfg.IS_DPN:
            for j in range(3):
                rim[j, :d1m, :d2m] = im[:d1m, :d2m, 2 - j] - pixel_means[2 - j]
        else:
            for j in range(3):
                rim[j, :d1m, :d2m] = (im[:d1m, :d2m, 2 - j] - pixel_means[2 - j]) * 0.0167
        if self.crop_size:
            return mx.nd.array(rim, dtype='float32')
        else:
            return mx.nd.array(rim, dtype='float32'), scale