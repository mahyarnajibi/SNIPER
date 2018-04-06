python main_chips.py
cp ~/CRCNN/output/chips_v4_1s/res50_coco_chips/train2014_val2014/CRCNN-0007.params ~/dc2/output/rcnn/cocof/resnet_v1_50_coco_rcnn_dcn_fast/train2014_val2014/rcnn-0004.params
cd ~/dc2
python run.py
cd ~/dc2/output/rcnn/cocof/resnet_v1_50_coco_rcnn_dcn_fast/test-dev2015/results
zip newsortneg.zip detections_test-dev2015_results.json
sudo shutdown -h
