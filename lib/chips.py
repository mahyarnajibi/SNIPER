from bbox.bbox_transform import clip_boxes, ignore_overlaps
import numpy as np

def genchipsones(width, height, boxes, chipsize):
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

    p = np.random.permutation(chips.shape[0])
    chips = chips[p]

    overlaps = ignore_overlaps(chips, boxes.astype(np.float))
    maxo = np.max(overlaps, axis=0)

    chip_matches = []
    num_matches = []
    for j in range(len(chips)):
        nvids = np.where(overlaps[j, :] == 1)[0].tolist()
        for k in range(len(boxes)):
            if overlaps[j,k] == maxo[k]:
                nvids.append(k)
        fvids = set(nvids)
        chip_matches.append(fvids)
        num_matches.append(len(fvids))

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


def genscorechips(width, height, boxes, chipsize, scores):
    chips = []
    boxes = clip_boxes(boxes, np.array([height-1, width-1]))
    # corners
    chips.append([max(width - chipsize, 0), 0, width - 1, min(chipsize, height-1)])
    chips.append([0, max(height - chipsize, 0), min(chipsize, width-1), height-1])
    chips.append([max(width - chipsize, 0), max(height - chipsize, 0), width-1, height-1])
    stride = np.random.randint(28,36)
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

    p = np.random.permutation(chips.shape[0])
    chips = chips[p]
    overlaps = ignore_overlaps(chips, boxes.astype(np.float))
    chip_matches = []
    score_matches = []
    for j in range(len(chips)):
        nvids = np.where(overlaps[j, :] > 0.8)[0]
        chip_matches.append(set(nvids.tolist()))
        score_matches.append(sum(scores[nvids]))

    fchips = []
    ct = 0
    while True:
        max_matches = max(score_matches)
        maxid = np.argmax(np.array(score_matches))
        if max_matches < 10:
            break
        bestchip = chip_matches[maxid]
        fchips.append(chips[maxid])
        ct = ct + 1
        # now remove all rois in bestchip
        for j in range(len(score_matches)):
            chip_matches[j] = chip_matches[j] - bestchip
            if len(chip_matches[j]) != 0:
                score_matches[j] = sum(scores[np.array(list(chip_matches[j]))])
            else:
                score_matches[j] = 0

    return fchips
