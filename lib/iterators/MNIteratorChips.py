import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import cv2
import numpy as np
from bbox.bbox_transform import clip_boxes
from bbox.bbox_regression import expand_bbox_regression_targets
from MNIteratorBase import MNIteratorBase
from bbox.bbox_transform import bbox_overlaps,bbox_pred
from bbox.bbox_regression import compute_bbox_regression_targets
def clip_boxes_with_chip(boxes,chip):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], chip[2] - 1), chip[0])
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], chip[3] - 1), chip[1])
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], chip[2] - 1), chip[0])
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], chip[3] - 1), chip[1])
    return boxes
class MNIteratorChips(MNIteratorBase):
    def __init__(self, roidb, config, batch_size = 4,  threads = 8, nGPUs = 1, pad_rois_to=400,single_size_change=False,crop_size=(400,400)):
        self.crop_size = crop_size
        self.num_classes = roidb[0]['gt_overlaps'].shape[1]
        self.bbox_means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (self.num_classes, 1))
        self.bbox_stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (self.num_classes, 1))
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        super(MNIteratorChips,self).__init__(roidb,config,batch_size,threads,nGPUs,pad_rois_to,single_size_change)
        
    def reset(self):
        #import pdb;pdb.set_trace()
        self.cur_i = 0
        n = len(self.roidb)

        nchips = np.zeros(n,dtype=int)
        for i,iroidb in enumerate(self.roidb):
            nchips[i] = len(iroidb['chips'])
        n_total_chips = nchips.sum()
        index = np.zeros(n_total_chips,dtype=int)
        counter = 0
        for i in range(n):
            index[counter:counter+nchips[i]] = i
            counter += nchips[i]
        # Make sure the index in divisable by batch size
        if index.shape[0]%self.batch_size>0:
            extra = self.batch_size - (index.shape[0]%self.batch_size)
            index = np.hstack((index,index[0:extra]))
        # Random permutation
        index = np.random.permutation(index)        

        self.inds = index
        self.size = len(self.inds)

        # Make chip counters zero
        # Assign boxes to chips
        print 'Assigning proposals to chips'
        for iroidb in self.roidb:
            iroidb['counter'] = 0
            valid_in_chips = iroidb['valid_in_chips']
            pid2chips = {}
            for chipi,valids in enumerate(valid_in_chips):
                for pid in valids:
                    if pid in pid2chips:
                        pid2chips[pid].append(chipi)
                    else:
                        pid2chips[pid] = [chipi]

            props_in_chip = [[] for _ in range(len(iroidb['chips']))]
            for pid in pid2chips:
                n_valid_chips = len(pid2chips[pid])
                sel_id = 0
                if n_valid_chips>1:
                    sel_id = np.random.randint(0,n_valid_chips)
                props_in_chip[pid2chips[pid][sel_id]].append(pid)
            iroidb['props_in_chip'] = props_in_chip
            iroidb['chip_order'] = np.random.permutation(np.arange(len(iroidb['chips'])))

        print 'Done!'


    def get_batch(self):
        if self.cur_i >= self.size:
            return False
        cur_roidbs = [self.roidb[self.inds[i%self.size]] for i in range(self.cur_i, self.cur_i+self.batch_size)]

        # Process cur roidb
        self.batch = self._get_batch(cur_roidbs)
        
        for c in cur_roidbs:
            c['counter']+= 1
        self.cur_i += self.batch_size
        return True

    def im_worker(self,data):
        im = data['im']
        crop_id = data['crop_id']
        roidb = data['roidb'].copy()
        crop = roidb['chips'][crop_id]
        
        # Crop the image
        im = im[int(crop[0][1]):int(crop[0][3]),int(crop[0][0]):int(crop[0][2]),:]

        # Scale the image
        crop_scale = crop[1]
        
        # Resize the crop
        im = cv2.resize(im, None, None, fx=crop_scale, fy=crop_scale, interpolation=cv2.INTER_LINEAR)
        # Check if we have less than the specified #pixels
        if(im.shape[0]>self.crop_size[1]):
            im = im[0:self.crop_size[1],:,:]
        if(im.shape[1]>self.crop_size[0]):
            im = im[:,0:self.crop_size[0],:]

        im_info = [im.shape[0],im.shape[1],crop_scale]

        #roidb['boxes'] = clip_boxes(np.round(roidb['boxes'] * im_scale), im_info[:2])
        roidb['im_info'] = im_info
        return {'im':im,'roidb': roidb}

    def visualize(self,im_tensor,boxes,labels,bbox_targets,bbox_weights):
        #import pdb;pdb.set_trace()
        im_tensor = im_tensor.asnumpy()
        for imi in range(im_tensor.shape[0]):
            im = np.zeros((im_tensor.shape[2],im_tensor.shape[3],3),dtype=np.uint8)
            for i in range(3):
                im[:,:,i] = im_tensor[imi,i,:,:] + self.pixel_mean[2 - i]
            # Visualize positives
            plt.imshow(im)
            pos_ids = np.where(labels[imi].asnumpy()>0)[0]
            cboxes = boxes[imi][pos_ids,1:5].asnumpy()

            for box in cboxes:
                    rect = plt.Rectangle((box[0], box[1]),
                                        box[2] - box[0],
                                        box[3] - box[1], fill=False,
                                        edgecolor='green', linewidth=3.5)
                    plt.gca().add_patch(rect)
            plt.savefig('debug/visualization/test_{}_pos.png'.format(imi))
            plt.cla()
            plt.clf()
            plt.close()
            # Visualize negatives
            plt.imshow(im)
            neg_ids = np.where(labels[imi].asnumpy()==0)[0]
            cboxes = boxes[imi][neg_ids,1:5].asnumpy()

            for box in cboxes:
                    rect = plt.Rectangle((box[0], box[1]),
                                        box[2] - box[0],
                                        box[3] - box[1], fill=False,
                                        edgecolor='red', linewidth=3.5)
                    plt.gca().add_patch(rect)
            plt.savefig('debug/visualization/test_{}_neg.png'.format(imi))
            plt.cla()
            plt.clf()
            plt.close()
            # Visualize bbox targets
            # bbox_pos_ids = np.where(bbox_weights[:,4]==1)[0]
            # cbbox_targets = bbox_targets[imi,bbox_pos_ids].asnumpy()
            # pos_rois = boxes[imi][bbox_pos_ids,1:5].asnumpy()
            # cbbox_targets[:, 4:] *= self.bbox_stds[1, :]
            
            # cboxes = bbox_pred(pos_rois,cbbox_targets)[:,4:]
            # plt.imshow(im)
            # for box in cboxes:
            #         rect = plt.Rectangle((box[0], box[1]),
            #                             box[2] - box[0],
            #                             box[3] - box[1], fill=False,
            #                             edgecolor='orange', linewidth=3.5)
            #         plt.gca().add_patch(rect)
            # plt.savefig('debug/visualization/test_{}_targets.png'.format(imi))
            # plt.cla()
            # plt.clf()
            # plt.close()

    def roidb_worker(self,data):

        roidb = data[0]
        im_i = data[1]
        im_info = roidb['im_info']
        # get rois
        chip_id = roidb['chip_order'][roidb['counter']%len(roidb['chips'])]
        cur_chip = roidb['chips'][chip_id][0]

        sel_box_ids = np.array(roidb['props_in_chip'][chip_id],dtype=int)
        # Select current boxes
        #sel_overlaps = roidb['gt_overlaps'][sel_box_ids]
        gt_overlaps = roidb['gt_overlaps']
        gt_classes = roidb['gt_classes']
        #sel_classes = roidb['gt_classes'][sel_box_ids]

        # Find all gts, not only those selected in the chip!
        # NOTE we do not crop gts!
        
        sel_gt_inds = np.where(gt_overlaps[sel_box_ids].max(1)==1)[0]
        sel_orig_gt_inds = sel_box_ids[sel_gt_inds]
        gt_inds = np.where(gt_overlaps.max(1)==1)[0]

        #import pdb;pdb.set_trace()
        sel_orig_gt_inds = np.searchsorted(gt_inds,sel_orig_gt_inds)
        gts = roidb['boxes'][gt_inds,:]

        # Compute overlaps in the original scale
        rois = roidb['boxes'][sel_box_ids,:]
        rois = clip_boxes_with_chip(rois,cur_chip)
        #import pdb;pdb.set_trace()

        
        overlaps = np.zeros((rois.shape[0], gt_overlaps.shape[1]), dtype=np.float32)

        if gts.shape[0]>0:
            new_gt_overlaps = bbox_overlaps(rois.astype(np.float), gts.astype(np.float))
            valid_bbox_ids = np.where(new_gt_overlaps[:,sel_orig_gt_inds].max(1)>= self.cfg.TRAIN.BBOX_REGRESSION_THRESH)
            argmaxes = new_gt_overlaps.argmax(axis=1)
            maxes = new_gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0] 
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]


        #gts = np.round(gts * im_info[2])

        #selected_gt_inds = np.where(gt_overlaps[sel_box_ids].max(1)==1)[0]

        #import pdb;pdb.set_trace()

        # Process boxes valid in the current gt
        

        # Translate boxes
        rois[:,0] -= cur_chip[0]
        rois[:,2] -= cur_chip[0]
        rois[:,1] -= cur_chip[1]
        rois[:,3] -= cur_chip[1]
        

        # Scale rois
        #rois = np.round(rois * im_info[2])
        # Clip boxes
        rois = clip_boxes(np.round(rois * im_info[2]), im_info[:2])

        
        #gt_inds = np.where(sel_overlaps.max(1)==1)[0]
        #gts = rois[gt_inds,:]
        # overlaps = np.zeros((rois.shape[0], gt_overlaps.shape[1]), dtype=np.float32)

        # if gts.shape[0]>0:
        #     new_gt_overlaps = bbox_overlaps(rois.astype(np.float), gts.astype(np.float))
        #     argmaxes = new_gt_overlaps.argmax(axis=1)
        #     maxes = new_gt_overlaps.max(axis=1)
        #     I = np.where(maxes > 0)[0] 
        #     overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        labels = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        # Make sure selected gts have an overlap of 1
        
        overlaps[sel_gt_inds] = 1

        num_classes = gt_overlaps.shape[1]
        # Compute bbox targets

        overlaps4bbox = np.zeros(overlaps.shape)
        labels4bbox = np.zeros(labels.shape)
        overlaps4bbox[valid_bbox_ids] = overlaps[valid_bbox_ids]
        labels4bbox[valid_bbox_ids] = labels[valid_bbox_ids]

        bbox_targets = compute_bbox_regression_targets(rois,overlaps4bbox,labels4bbox,self.cfg)
        bbox_num_classes = 2 if self.cfg.CLASS_AGNOSTIC else self.num_classes

        for cls in range(1, bbox_num_classes):
            cls_indexes = np.where(bbox_targets[:, 0] > 0) if self.cfg.CLASS_AGNOSTIC else np.where(bbox_targets[:, 0] == cls)[0]
            bbox_targets[cls_indexes, 1:] -= self.bbox_means[cls, :]
            bbox_targets[cls_indexes, 1:] /= self.bbox_stds[cls, :]


        # Since we have ohem pass all rois
        rois_per_image = rois.shape[0]
        fg_rois_per_image = rois_per_image
        im_rois, labels, bbox_targets, bbox_weights = \
            self.sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
                            labels, overlaps, bbox_targets)
        rois = im_rois

        if rois.shape[0] > self.n_expected_roi:
            rois = rois[0:self.n_expected_roi, :]
            bbox_weights = bbox_weights[0:self.n_expected_roi, :]
            bbox_targets = bbox_targets[0:self.n_expected_roi, :]
            labels = labels[0:self.n_expected_roi]
        elif rois.shape[0] < self.n_expected_roi:
            n_pad = self.n_expected_roi - rois.shape[0]
            rois = np.vstack((rois, np.repeat(self.pad_roi, n_pad, axis=0)))
            labels = np.hstack((labels, np.repeat(self.pad_label, n_pad, axis=0)))
            bbox_weights = np.vstack((bbox_weights, np.repeat(self.pad_weights, n_pad, axis=0)))
            bbox_targets = np.vstack((bbox_targets, np.repeat(self.pad_targets, n_pad, axis=0)))

        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))

        return {'rois':rois_array_this_image,'labels':labels,
        'bbox_weights':bbox_weights,'bbox_targets':bbox_targets}



    def _get_batch(self,roidb):
        """
        return a dict of multiple images
        :param roidb: a list of dict, whose length controls batch size
        ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
        :return: data, label
        """
        num_images = len(roidb)
        im_tensor, roidb = self.im_process(roidb)
        assert self.cfg.TRAIN.BATCH_ROIS == -1 or self.cfg.TRAIN.BATCH_ROIS % self.cfg.TRAIN.BATCH_IMAGES == 0, \
            'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(self.cfg.TRAIN.BATCH_IMAGES, self.cfg.TRAIN.BATCH_ROIS)

        worker_data = [(roidb[i],i%self.n_per_gpu) for i in range(len(roidb))]   
        # all_labels = []
        # for cdata in worker_data:
        #     all_labels.append(self.roidb_worker(cdata))

        all_labels = self.thread_pool.map(self.roidb_worker,worker_data)
        
        rois = mx.nd.zeros((num_images,self.n_expected_roi,5),mx.cpu(0))
        labels = mx.nd.zeros((num_images,self.n_expected_roi),mx.cpu(0))
        bbox_targets = mx.nd.zeros((num_images,self.n_expected_roi,8),mx.cpu(0))
        bbox_weights = mx.nd.zeros((num_images,self.n_expected_roi,8),mx.cpu(0))
        for i,clabel in enumerate(all_labels):
            rois[i] = clabel['rois']
            labels[i] = clabel['labels']
            bbox_targets[i] = clabel['bbox_targets']
            bbox_weights[i] = clabel['bbox_weights']
        #if self.cur_i==0:
        #self.visualize(im_tensor,rois,labels,bbox_targets,bbox_weights)
        #import pdb;pdb.set_trace()
        self.data = [im_tensor,rois]
        self.label = [labels,bbox_targets,bbox_weights]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(), provide_data=self.provide_data, provide_label=self.provide_label)



    def im_process(self,roidb):    
        n_batch = len(roidb)
        ims = []
        for i in range(n_batch):
            crop_id = roidb[i]['chip_order'][roidb[i]['counter']%len(roidb[i]['chips'])]
            im = cv2.imread(roidb[i]['image'], cv2.IMREAD_COLOR)
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            ims.append({'im':im,'crop_id':crop_id,'roidb':roidb[i]})
        processed_list = self.thread_pool.map(self.im_worker,ims)
        processed_roidb = [p['roidb'] for p in processed_list]

        im_tensor = mx.ndarray.zeros((n_batch,
            3,self.crop_size[0],self.crop_size[1]))
        #print im_tensor.shape
        for i in range(len(processed_list)):
            im = processed_list[i]['im']
            for j in range(3):
                im_tensor[i, j, 0:im.shape[0], 0:im.shape[1]] = im[:, :, 2 - j] - self.pixel_mean[2 - j]
        return im_tensor,processed_roidb


    def sample_rois(self,rois, fg_rois_per_image, rois_per_image, num_classes,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
        """
        generate random sample of ROIs comprising foreground and background examples
        :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
        :param fg_rois_per_image: foreground roi number
        :param rois_per_image: total roi number
        :param num_classes: number of classes
        :param labels: maybe precomputed
        :param overlaps: maybe precomputed (max_overlaps)
        :param bbox_targets: maybe precomputed
        :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
        :return: (labels, rois, bbox_targets, bbox_weights)
        """
        cfg = self.cfg
        if labels is None:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

        # foreground RoI with FG_THRESH overlap
        fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
        # Sample foreground regions without replacement
        if len(fg_indexes) > fg_rois_per_this_image:
            fg_indexes = np.random.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
        # Sample foreground regions without replacement
        if len(bg_indexes) > bg_rois_per_this_image:
            bg_indexes = np.random.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

        # indexes selected
        keep_indexes = np.append(fg_indexes, bg_indexes)

        # pad more to ensure a fixed minibatch size
        while keep_indexes.shape[0] < rois_per_image:
            gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
            gap_indexes = np.random.choice(range(len(rois)), size=gap, replace=False)
            keep_indexes = np.append(keep_indexes, gap_indexes)

        # select labels
        labels = labels[keep_indexes]
        # set labels of bg_rois to be 0
        labels[fg_rois_per_this_image:] = 0
        rois = rois[keep_indexes]

        # load or compute bbox_target
        if bbox_targets is not None:
            bbox_target_data = bbox_targets[keep_indexes, :]
        else:
            targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                           / np.array(cfg.TRAIN.BBOX_STDS))
            bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

        bbox_targets, bbox_weights = \
            expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

        return rois, labels, bbox_targets, bbox_weights

    

