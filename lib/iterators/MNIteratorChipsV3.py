import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import cv2
import numpy as np
from bbox.bbox_regression import expand_bbox_regression_targets
from MNIteratorBase import MNIteratorBase
from bbox.bbox_transform import bbox_overlaps,bbox_pred,bbox_transform,clip_boxes, filter_boxes, ignore_overlaps
from bbox.bbox_regression import compute_bbox_regression_targets
from chips import genchips
from multiprocessing import Pool
import time
from HelperV3 import im_worker,roidb_worker
def clip_boxes_with_chip(boxes,chip):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], chip[2] - 1), chip[0])
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], chip[3] - 1), chip[1])
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], chip[2] - 1), chip[0])
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], chip[3] - 1), chip[1])
    return boxes

def chip_worker(r):
    width = r['width']
    height = r['height']
    im_size_max = max(width, height)
    im_scale_1 = 3
    im_scale_2 = 1.667
    im_scale_3 = 512.0 / float(im_size_max)

    gt_boxes = r['boxes'][np.where(r['max_overlaps'] == 1)[0], :]

    # for chip in gt_boxes:
    #     cv2.rectangle(im, (int(chip[0]), int(chip[1])), (int(chip[2]), int(chip[3])),
    #     (0, 0, 255), 3)
    #     cv2.imshow('image', im)
    ws = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(np.int32)
    hs = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(np.int32)
    area = np.sqrt(ws*hs)
    ms = np.maximum(ws,hs)

    ids1 = np.where((ms < 80) & (ws >= 10) & (hs>=10))[0]
    ids2 = np.where((ms >= 32) & (ms < 150)& (ws >= 10) & (hs>=10))[0]
    ids3 = np.where((ms >= 120) & (ws >= 10) & (hs>=10))[0]

    chips1 = genchips(int(r['width'] * im_scale_1), int(r['height'] * im_scale_1), gt_boxes[ids1, :] * im_scale_1, 512)
    chips2 = genchips(int(r['width'] * im_scale_2), int(r['height'] * im_scale_2), gt_boxes[ids2, :] * im_scale_2, 512)
    chips3 = genchips(int(r['width'] * im_scale_3), int(r['height'] * im_scale_3), gt_boxes[ids3, :] * im_scale_3, 512)
    chips1 = np.array(chips1)/im_scale_1
    chips2 = np.array(chips2)/im_scale_2
    chips3 = np.array(chips3)/im_scale_3

    chip_ar = []
    for chip in chips1:
        chip_ar.append([chip,3])
    for chip in chips2:
        chip_ar.append([chip,1.667])
    for chip in chips3:
        chip_ar.append([chip,im_scale_3])


    return chip_ar

    #return (np.array(chips1),np.array(chips2).np.array(chips3),im_scale_3)

def props_in_chip_worker(r):
    props_in_chips = [[] for _ in range(len(r['crops']))]
    widths = (r['boxes'][:,2] - r['boxes'][:,0]).astype(np.int32)
    heights= (r['boxes'][:,3] - r['boxes'][:,1]).astype(np.int32)
    max_sizes = np.maximum(widths,heights)
    area = np.sqrt(widths*heights)
    sids = np.where((max_sizes <80) & (widths >= 10) & (heights>=10))[0]
    mids = np.where((max_sizes>=32) & (max_sizes < 150)& (widths >= 10) & (heights>=10))[0]
    bids = np.where((max_sizes>=120) & (widths >= 10) & (heights>=10))[0]
    
    chips1,chips2,chips3 = [],[],[]
    chip_ids1, chip_ids2, chip_ids3  = [],[],[]
    for ci,crop in enumerate(r['crops']):
        if crop[1]==3:
            chips1.append(crop[0])
            chip_ids1.append(ci)
        elif crop[1]==1.667:
            chips2.append(crop[0])
            chip_ids2.append(ci)
        else:
            chips3.append(crop[0])
            chip_ids3.append(ci)

    chips1 = np.array(chips1,dtype=np.float)
    chips2 = np.array(chips2,dtype=np.float)
    chips3 = np.array(chips3,dtype=np.float)
    chip_ids1 = np.array(chip_ids1)
    chip_ids2 = np.array(chip_ids2)
    chip_ids3 = np.array(chip_ids3)

    small_boxes = r['boxes'][sids].astype(np.float)
    med_boxes = r['boxes'][mids].astype(np.float)
    big_boxes = r['boxes'][bids].astype(np.float)

    if chips1.shape[0]>0:
        overlaps = ignore_overlaps(chips1,small_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi,cid in enumerate(max_ids):
            props_in_chips[chip_ids1[cid]].append(sids[pi])

    if chips2.shape[0]>0:
        overlaps = ignore_overlaps(chips2,med_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi,cid in enumerate(max_ids):
            props_in_chips[chip_ids2[cid]].append(mids[pi])
    if chips3.shape[0]>0:
        overlaps = ignore_overlaps(chips3,big_boxes)
        max_ids = overlaps.argmax(axis=0)
        for pi,cid in enumerate(max_ids):
            props_in_chips[chip_ids3[cid]].append(bids[pi])

    for j in range(len(props_in_chips)):
        props_in_chips[j] = np.array(props_in_chips[j],dtype=np.int32)

    return props_in_chips


class MNIteratorChips(MNIteratorBase):
    def __init__(self, roidb, config, batch_size = 4,  threads = 8, nGPUs = 1, pad_rois_to=400,crop_size=(512,512)):
        self.crop_size = crop_size
        self.num_classes = roidb[0]['gt_overlaps'].shape[1]
        self.bbox_means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (self.num_classes, 1))
        self.bbox_stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (self.num_classes, 1))
        self.data_name = ['data',  'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.pool = Pool(8)
        self.context_size = 320
        super(MNIteratorChips,self).__init__(roidb,config,batch_size,threads,nGPUs,pad_rois_to,False)
    


    def reset(self):
        self.cur_i = 0

        self.crop_idx = [0] * len(self.roidb)

        chips = self.pool.map(chip_worker,self.roidb)

        chipindex = []
        chip_count =0
        for i,r in enumerate(self.roidb):
            cs = chips[i]
            chip_count += len(cs)
            r['crops'] = cs
            for j in range(len(cs)):
                chipindex.append(i)

        print('N chips: {}'.format(chip_count))

        all_props_in_chips = self.pool.map(props_in_chip_worker,self.roidb)
        for props_in_chips,cur_roidb in zip(all_props_in_chips,self.roidb):
            cur_roidb['props_in_chips'] = props_in_chips

        blocksize = self.batch_size
        chipindex = np.array(chipindex)
        if chipindex.shape[0] % blocksize > 0:
            extra = blocksize - (chipindex.shape[0] % blocksize)
            chipindex = np.hstack((chipindex, chipindex[0:extra]))
        allinds = np.random.permutation(chipindex)
        avg = 0

        # Distribute based on avg
        tmp_crop_idx = [0] * len(self.roidb)
        gpu_inds = np.reshape(allinds,(-1,self.n_per_gpu))
        neach_chip = np.zeros_like(gpu_inds,dtype=int)
        nper_gpu = np.zeros(gpu_inds.shape[0],dtype=int)
        chipid2imchip = {}
        imchip2chipid = {}
        count = 0
        allchipinds = np.zeros_like(allinds,dtype=int)
          
        for i in allinds:
            cur_gpu_ind = count / self.n_per_gpu
            n_chip = self.roidb[i]['props_in_chips'][tmp_crop_idx[i]].shape[0]
            neach_chip[cur_gpu_ind,count%self.n_per_gpu] = n_chip
            gpu_inds[cur_gpu_ind,count%self.n_per_gpu] = count
            nper_gpu[cur_gpu_ind] += n_chip
            chipid2imchip[count] = (i,tmp_crop_idx[i])
            imchip2chipid[(i,tmp_crop_idx[i])] = count
            tmp_crop_idx[i] = (tmp_crop_idx[i]+1) % len(self.roidb[i]['crops'])
            count+=1
        print ('Standard Deviation Before Balancing: {}'.format(nper_gpu.std()))
        avg = int(nper_gpu.mean())
        for i,n in enumerate(nper_gpu):
            if n<avg: continue
            while n>avg:
                # which index to remove?
                diffs = np.abs(avg - (n-neach_chip[i]))
                rm_ind = diffs.argmin()

                n_after_rm = avg - (nper_gpu[i] - neach_chip[i][rm_ind])
                # if(n_after_rm > nper_gpu[i]-avg):
                #     break

                # with whom we should replace this?
                diffs = avg-nper_gpu
                sel_gpu = diffs.argmax()
                sel_chip = neach_chip[sel_gpu].argmin()
                if neach_chip[sel_gpu][sel_chip]>= neach_chip[i][rm_ind]:
                    break
                # Update sum arrays
                nper_gpu[i] = nper_gpu[i] - neach_chip[i][rm_ind] + neach_chip[sel_gpu][sel_chip]
                nper_gpu[sel_gpu] = nper_gpu[sel_gpu] - neach_chip[sel_gpu][sel_chip] + neach_chip[i][rm_ind]
                tmp = neach_chip[i][rm_ind]
                neach_chip[i][rm_ind] = neach_chip[sel_gpu][sel_chip]
                neach_chip[sel_gpu][sel_chip] = tmp
                n = nper_gpu[i]

                # Update gpu inds
                tmp_id = gpu_inds[sel_gpu][sel_chip]
                gpu_inds[sel_gpu][sel_chip] = gpu_inds[i][rm_ind]
                gpu_inds[i][rm_ind] = tmp_id

        print('Standard Deviation After Balancing: {}'.format(nper_gpu.std()))
        allchipinds = gpu_inds.reshape(-1)
        for r in self.roidb:
            r['chip_order'] = []
        allinds = []
        # Assigning chip order to roidb



        for ind in allchipinds:
            roidb_ind, chip_ind = chipid2imchip[ind]
            allinds.append(roidb_ind)
            self.roidb[roidb_ind]['chip_order'].append(chip_ind)
        avg = nper_gpu.mean()
        print('Avg per GPU: {}'.format(avg))
        per_im_avg = float(nper_gpu.sum())/(len(allinds))
        print('Avg per chip: {}'.format(per_im_avg))


        self.inds = np.array(allinds,dtype=int)
           
        # nper_gpu = np.zeros(gpu_inds.shape[0],dtype=int)
        # tmp_crop_idx = [0] * len(self.roidb)
        # count = 0;
        # for ind in self.inds:
        #     chip_id = self.roidb[ind]['chip_order'][tmp_crop_idx[ind]]
        #     n = self.roidb[ind]['props_in_chips'][chip_id].shape[0]
        #     tmp_crop_idx[ind] += 1
        #     nper_gpu[count/self.n_per_gpu] += n
        #     count += 1
        # print nper_gpu
                
        self.size = len(self.inds)
        print 'Done!'


    def get_batch(self):
        if self.cur_i >= self.size:
            return False

        #cur_roidbs = [self.roidb[self.inds[i%self.size]] for i in range(self.cur_i, self.cur_i+self.batch_size)]

        # Process cur roidb
        self.batch = self._get_batch()
        
        self.cur_i += self.batch_size
        return True

    def _get_batch(self):
        """
        return a dict of multiple images
        :param roidb: a list of dict, whose length controls batch size
        ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
        :return: data, label
        """
        

        cur_from = self.cur_i
        cur_to = self.cur_i+self.batch_size
        roidb = [self.roidb[self.inds[i]] for i in range(cur_from, cur_to)]
        #num_images = len(roidb)
        cropids = [self.roidb[self.inds[i]]['chip_order'][self.crop_idx[self.inds[i]]] for i in range(cur_from, cur_to)]
        n_batch = len(roidb)
        ims = []
        for i in range(n_batch):            
            ims.append([roidb[i]['image'], roidb[i]['crops'][cropids[i]],roidb[i]['flipped'], self.crop_size[0]])
        

        for i in range(cur_from, cur_to):
            self.crop_idx[self.inds[i]] = self.crop_idx[self.inds[i]] + 1

        processed_list = self.thread_pool.map(im_worker,ims)
        #im_tensor, roidb = self.im_process(roidb,cropids)
        processed_roidb = []
        t1 = time.time()        
        for i in range(len(roidb)):
            tmp = roidb[i].copy()
            scale = roidb[i]['crops'][cropids[i]][1]
            tmp['im_info'] = [self.crop_size[0],self.crop_size[1],scale]
            processed_roidb.append(tmp)
        im_tensor = mx.nd.zeros((n_batch,3,self.crop_size[0],self.crop_size[1]), dtype=np.float32)
        for i in range(len(processed_list)):
            im_tensor[i, :, :, :] = processed_list[i]

        # assert self.cfg.TRAIN.BATCH_ROIS == -1 or self.cfg.TRAIN.BATCH_ROIS % self.cfg.TRAIN.BATCH_IMAGES == 0, \
        #     'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(self.cfg.TRAIN.BATCH_IMAGES, self.cfg.TRAIN.BATCH_ROIS)

        #worker_data = [(processed_roidb[i],i%self.n_per_gpu,cropids[i]) for i in range(n_batch)]   
        # all_labels = []
        # for cdata in worker_data:
        #     all_labels.append(self.roidb_worker(cdata))
        worker_data = []
        for i in range(len(processed_roidb)):
            cropid = cropids[i]
            tmp = processed_roidb[i]['crops'][cropid][0]
            nids = processed_roidb[i]['props_in_chips'][cropid]
            gtids = np.where(processed_roidb[i]['max_overlaps'] == 1)[0]
            gt_boxes = processed_roidb[i]['boxes'][gtids, :]
            gt_classes = processed_roidb[i]['max_classes'][gtids]
            boxes = processed_roidb[i]['boxes'].copy()
            cur_crop = processed_roidb[i]['crops'][cropid][0]
            im_scale = processed_roidb[i]['crops'][cropid][1]

            argw = [i%self.n_per_gpu, processed_roidb[i]['im_info'], cur_crop, im_scale, nids, gt_boxes, gt_classes, boxes, self.n_expected_roi, self.num_classes]
            worker_data.append(argw)

        #all_labels = self.thread_pool.map(self.roidb_worker,worker_data)
        all_labels = self.pool.map(roidb_worker,worker_data)

        rois = mx.nd.zeros((n_batch,self.n_expected_roi,5),mx.cpu(0))
        labels = mx.nd.zeros((n_batch,self.n_expected_roi),mx.cpu(0))
        bbox_targets = mx.nd.zeros((n_batch,self.n_expected_roi,8),mx.cpu(0))
        bbox_weights = mx.nd.zeros((n_batch,self.n_expected_roi,8),mx.cpu(0))
        for i in range(len(all_labels)):
            rois[i] = all_labels[i][0]
            labels[i] = all_labels[i][1]
            bbox_targets[i] = all_labels[i][2]
            bbox_weights[i] = all_labels[i][3]

        #if self.cur_i==0:

        # boxes= [p['gt_boxes'] for p in all_labels]
        # import pdb;pdb.set_trace()
        #self.visualize(im_tensor,rois,labels,None,None)
        #import pdb;pdb.set_trace()
        self.data = [im_tensor, rois]
        self.label = [labels,bbox_targets,bbox_weights]
        return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(), provide_data=self.provide_data, provide_label=self.provide_label)



    # def im_worker(self,data):
    #     im = data['im']
    #     crop_id = data['crop_id']
    #     roidb = data['roidb'].copy()
    #     crop = roidb['crops'][crop_id]
        
    #     # Crop the image
    #     origim = im[int(crop[0][1]):int(crop[0][3]),int(crop[0][0]):int(crop[0][2]),:]
    #     # Scale the image
    #     crop_scale = crop[1]
        
    #     # Resize the crop
    #     if int(origim.shape[0]*0.625)==0 or int(origim.shape[1]*0.625)==0:
    #         print 'Something wrong3'
    #     try:
    #         im = cv2.resize(origim, None, None, fx=crop_scale, fy=crop_scale, interpolation=cv2.INTER_LINEAR)
    #         im_context = cv2.resize(origim, None, None, fx=crop_scale*0.625, fy=crop_scale*0.625, interpolation=cv2.INTER_LINEAR)
    #     except:
    #         print 'Something wrong4'
        

    #     # Check if we have less than the specified #pixels
    #     if(im.shape[0]>self.crop_size[1]):
    #         im = im[0:self.crop_size[1],:,:]
    #     if(im.shape[1]>self.crop_size[0]):
    #         im = im[:,0:self.crop_size[0],:]

    #     if(im_context.shape[0]>self.context_size):
    #         im_context = im_context[0:self.context_size,:,:]
    #     if(im_context.shape[1]>self.context_size):
    #         im_context = im_context[:,0:self.context_size,:]

    #     im_info = [im.shape[0],im.shape[1],crop_scale]

    #     roidb['im_info'] = im_info
    #     return {'im':im, 'im_context': im_context,'roidb': roidb}


    def roidb_worker(self,data):
        #import pdb;pdb.set_trace()

        roidb = data[0]
        new_rec = roidb.copy()
        im_i = data[1]
        cropid = data[2]

        im_info = roidb['im_info']
        # get rois
        cur_crop = roidb['crops'][cropid][0]
        im_scale = roidb['crops'][cropid][1]
        
        nids = roidb['props_in_chips'][cropid]

        gtids = np.where(new_rec['max_overlaps'] == 1)[0]

        gt_boxes = new_rec['boxes'][gtids, :].copy()
        gt_labs = new_rec['max_classes'][gtids]


        gt_boxes[:, 0] = gt_boxes[:, 0] - cur_crop[0]
        gt_boxes[:, 2] = gt_boxes[:, 2] - cur_crop[0]
        gt_boxes[:, 1] = gt_boxes[:, 1] - cur_crop[1]
        gt_boxes[:, 3] = gt_boxes[:, 3] - cur_crop[1]

        gt_boxes = clip_boxes(np.round(gt_boxes * im_scale), im_info[:2])
        ids = filter_boxes(gt_boxes, 10)
        if len(ids)>0:
            gt_boxes = gt_boxes[ids]
            gt_labs = gt_labs[ids]

        gt_boxes = np.hstack((gt_boxes, gt_labs.reshape(len(gt_labs), 1)))
        
        crois = new_rec['boxes'].copy()
        
        crois[:, 0] = crois[:, 0] - cur_crop[0]
        crois[:, 2] = crois[:, 2] - cur_crop[0]
        crois[:, 1] = crois[:, 1] - cur_crop[1]
        crois[:, 3] = crois[:, 3] - cur_crop[1]

        
        new_rec['boxes'] = clip_boxes(np.round(crois * im_scale), im_info[:2])

        ids = filter_boxes(new_rec['boxes'], 10)
        tids = np.intersect1d(ids, nids)
        if len(nids) > 0:
            ids = tids
        else:
            ids = nids

        if len(ids) > 0:            
            new_rec['boxes'] = new_rec['boxes'][ids, :]
            new_rec['max_overlaps'] = new_rec['max_overlaps'][ids]
            new_rec['max_classes'] = new_rec['max_classes'][ids]
            new_rec['bbox_targets'] = new_rec['bbox_targets'][ids, :]
            new_rec['gt_overlaps'] = new_rec['gt_overlaps'][ids]


        new_rec['im_info'] = im_info
        new_rec['gt_boxes'] = gt_boxes

        # infer num_classes from gt_overlaps
        num_classes = new_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = new_rec['boxes']
        gt_boxes = new_rec['gt_boxes']

        overlaps = ignore_overlaps(rois.astype(np.float), gt_boxes.astype(np.float))
        mov = np.max(overlaps)

        if mov < 1:
            print 'Something Wrong 1'
            import pdb;pdb.set_trace()
        fg_rois_per_image = len(rois)
        rois_per_image = fg_rois_per_image
        rois, labels, bbox_targets, bbox_weights = self.sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, self.cfg, gt_boxes=gt_boxes)


        
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
        if rois_array_this_image.shape[0]==0:
            print 'Something Wrong2'
        return {'rois':rois_array_this_image,'labels':labels,
        'bbox_weights':bbox_weights,'bbox_targets':bbox_targets,'gt_boxes':gt_boxes}


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
            #cboxes = boxes[imi][:, 0:4]
            for box in cboxes:
                    rect = plt.Rectangle((box[0], box[1]),
                                        box[2] - box[0],
                                        box[3] - box[1], fill=False,
                                        edgecolor='green', linewidth=3.5)
                    plt.gca().add_patch(rect)
            num = np.random.randint(100000)
            plt.savefig('debug/visualization/test_{}_pos.png'.format(num))
            plt.cla()
            plt.clf()
            plt.close()
            # # Visualize negatives
            # plt.imshow(im)
            # neg_ids = np.where(labels[imi].asnumpy()==0)[0]
            # cboxes = boxes[imi][neg_ids,1:5].asnumpy()
            #
            # for box in cboxes:
            #         rect = plt.Rectangle((box[0], box[1]),
            #                             box[2] - box[0],
            #                             box[3] - box[1], fill=False,
            #                             edgecolor='red', linewidth=3.5)
            #         plt.gca().add_patch(rect)
            # plt.savefig('debug/visualization/test_{}_neg.png'.format(imi))
            # plt.cla()
            # plt.clf()
            # plt.close()
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




    

    
    def sample_rois(self,rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, scale_sd=1):
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

        if labels is None:
            #overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float)
            overlaps = bbox_overlaps(rois.astype(np.float), gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

        """if len(gt_boxes) > 0:
            overlaps = bbox_overlaps(rois.astype(np.float), gt_boxes[:, :4].astype(np.float))
            rgt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            inv_inds = np.where(overlaps < 0.5)[0]
            rgt_assignment[inv_inds] = -1
            import pdb
            pdb.set_trace()
        else:
            rgt_assignment = -np.ones((len(rois)))"""
            
        
        
        thresh = 0.5
        # foreground RoI with FG_THRESH overlap
        if scale_sd == 0.5:
            thresh = 0.6

        if scale_sd == 0.25:
            thresh = 0.7    

        fg_indexes = np.where(overlaps >= thresh)[0]
        
        # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
        # Sample foreground regions without replacement
        if len(fg_indexes) > fg_rois_per_this_image:
            fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
        # Sample foreground regions without replacement
        if len(bg_indexes) > bg_rois_per_this_image:
            bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

        # indexes selected
        keep_indexes = np.append(fg_indexes, bg_indexes)

        # pad more to ensure a fixed minibatch size
        while keep_indexes.shape[0] < rois_per_image:
            gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
            gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
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
            #targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
            targets = bbox_transform(rois, gt_boxes[gt_assignment[keep_indexes], :4])
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                           / (np.array(cfg.TRAIN.BBOX_STDS) * scale_sd ))
            bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

        bbox_targets, bbox_weights = \
            expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

        return rois,  labels, bbox_targets, bbox_weights