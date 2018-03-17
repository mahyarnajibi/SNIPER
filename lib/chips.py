from bbox.bbox_transform import clip_boxes, ignore_overlaps
import numpy as np

def genchips(width, height, boxes, chipsize):
    chips = []
    boxes = clip_boxes(boxes, np.array([height-1, width-1]))
    # ensure coverage of image for worst case

    # corners
    chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
    chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
    chips.append([max(width - chipsize, 0), max(height - chipsize, 0), width-1, height-1])
    stride = 32
    for i in range(0, width - int(chipsize), stride):
        for j in range(0, height - int(chipsize), stride):
            x1 = i
            y1 = j
            x2 = i + chipsize - 1
            y2 = j + chipsize - 1
            chips.append([x1, y1, x2, y2])

    for j in range(0, height - int(chipsize), stride):
        x1 = max(width - chipsize - 1,0)
        y1 = j
        x2 = width - 1
        y2 = j + chipsize - 1
        chips.append([x1, y1, x2, y2])

    for i in range(0, width - int(chipsize), stride):
        x1 = i
        y1 = max(height - chipsize - 1,0)
        x2 = i + chipsize - 1
        y2 = height - 1
        chips.append([x1, y1, x2, y2])

    chips = np.array(chips).astype(np.float)

    """for i in range(len(chips)):
        if chips[i][0] < 0:
            chips[i][0] = 0
        if chips[i][1] < 0:
            chips[i][1] = 0

        if chips[i][2] >= width:
            chips[i][2] = width - 1
        if chips[i][3] >= height:
            chips[i][3] = height - 1"""

    p = np.random.permutation(chips.shape[0])
    chips = chips[p]

    overlaps = ignore_overlaps(chips, boxes.astype(np.float))
    maxo = np.max(overlaps, axis=0)
    missing = np.where(maxo < 1)[0]
    if len(missing) > 0:
        print('bulba')
        assert False, 'bulba'

    chip_matches = []
    num_matches = []
    for j in range(len(chips)):
        #nvids = np.where(overlaps[j, :] > 0.9)[0]
        nvids = np.where(overlaps[j, :] == 1)[0]
        chip_matches.append(set(nvids.tolist()))
        num_matches.append(len(nvids))

    fchips = []
    totalmatches = 0
    while True:
        max_matches = 0
        max_match = max(num_matches)
        mid = np.argmax(np.array(num_matches))
        if max_match == 0:
            break
        if max_match > max_matches:
            max_matches = max_match
            maxid = mid
        bestchip = chip_matches[maxid]
        fchips.append(chips[maxid])
        totalmatches = totalmatches + max_matches

        # now remove all rois in bestchip
        for j in range(len(num_matches)):
            chip_matches[j] = chip_matches[j] - bestchip
            num_matches[j] = len(chip_matches[j])

    return fchips