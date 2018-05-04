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

    #missing = np.where(maxo < 1)[0]
    #if len(missing) > 0:
    #    print('bulba')
    #    assert False, 'bulba'

    chip_matches = []
    num_matches = []
    for j in range(len(chips)):
        #nvids = np.where(overlaps[j, :] > 0.9)[0]
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
    stride = np.random.randint(52,60)    
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

tv1 = 0
tv2 = 0
tv3 = 0
tv4 = 0
act = 1

def genscorechips(width, height, boxes, chipsize, scores):
    import time
    #global tv1
    #global tv2
    #global tv3
    #global tv4
    #global act
    #t1 = time.time()
    chips = []
    boxes = clip_boxes(boxes, np.array([height-1, width-1]))

    # ensure coverage of image for worst case

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
    #t2 = time.time() - t1
    #tv1 = tv1 + t2
    t2f = time.time()
    #print ('init time ' + str(tv1/act))
    overlaps = ignore_overlaps(chips, boxes.astype(np.float))
    t2 = time.time() - t2f
    #tv2 = tv2 + t2
    #print ('overlap time ' + str(tv2/act))
    t2f = time.time()
    chip_matches = []
    score_matches = []
    num_matches = []
    for j in range(len(chips)):
        nvids = np.where(overlaps[j, :] > 0.8)[0]
        chip_matches.append(set(nvids.tolist()))
        score_matches.append(sum(scores[nvids]))
        num_matches.append(len(nvids))
    t2 = time.time() - t2f
    #tv3 = tv3 + t2
    #print ('match time ' + str(tv3/act))
    #t2f = time.time()
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
    #t2 = time.time() - t2f
    #tv4 = tv4 + t2
    #print ('generate time ' + str(tv4/act))
    #print ('total time ' + str((tv1+tv2+tv3+tv4)/act))
    #act = act + 1
    return fchips
