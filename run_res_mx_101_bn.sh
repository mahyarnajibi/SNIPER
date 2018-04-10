python main_chips_res_mx_101.py --cfg configs/faster/res101_mx_bn1.yml
cp ~/CRCNN/output/chips_resnet101_mx_bn/res101_mx_bn1/train2014_val2014/CRCNNv2-0004.params ~/CRCNN/data/pretrained_model/
python main_chips_res_mx_101.py --momentum 0.995 --cfg configs/faster/res101_mx_bn2.yml
cp ~/CRCNN/output/chips_resnet101_mx_bn/res101_mx_bn2/train2014_val2014/CRCNNv2-0002.params ~/CRCNN/data/pretrained_model/
python main_chips_res_mx_101.py --momentum 0.999 --cfg configs/faster/res101_mx_bn3.yml
sudo shutdown -h