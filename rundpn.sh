python main_chips.py
cp ~/CRCNN/output/chips_v4/dpn98_coco_chips/train2014_val2014/CRCNN-0004.params ~/CRCNN/data/pretrained_model/
python main_chips.py --momentum 0.995 --cfg configs/faster/dpn98_coco_chips_l2.yml
cp ~/CRCNN/output/chips_v4/dpn98_coco_chips_l2/train2014_val2014/CRCNN-0002.params ~/CRCNN/data/pretrained_model/
python main_chips.py --momentum 0.999 --cfg configs/faster/dpn98_coco_chips_l3.yml
#cp ~/CRCNN/output/chips_v4/res50_coco_chips/train2014_val2014/CRCNN-0001.params ~/dc2/output/rcnn/cocof/symbol_dpn_98_cls_coco_rcnn_dcn_fast/train2014_val2014/rcnn-0001.params
#cd ~/dc2
#python run.py
#cd ~/dc2/output/rcnn/cocof/resnet_v1_50_coco_rcnn_dcn_fast/test-dev2015/results
#zip newsortneg.zip detections_test-dev2015_results.json
sudo shutdown -h
