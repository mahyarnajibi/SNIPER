import pickle
import os

def vis_boxes(file_name, im_array, detections, scale, cfg, threshold, class_names_matching):
    """
    visualize detections in one image and save it
    :param file_name: file name of the output image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param scale: visualize the scaled image
    :param cfg: config
    :return:
    """
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import random
    im = im_array
    plt.imshow(im)

    for j in range(len(detections)):
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue

            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)

            plt.gca().text(bbox[0], bbox[1],
                           '{:s} {:.2f}'.format(class_names_matching[j], score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=16, color='white')

    
    vis_out_path = "./vis_result"

    file_name = (file_name.split('/')[-1]).split('.')[0]
    plt.savefig(os.path.join(vis_out_path, file_name + '.png'))
