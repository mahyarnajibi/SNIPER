# R-FCN-3000 at 30fps: Decoupling Detection and Classification

<p align="center">
<img src="http://legacydirs.umiacs.umd.edu/~najibi/github_readme_files/rfcn_3k.png" />
 </p>

R-FCN-3k is a real-time detector for up to 3,130 classes. The idea is to decouple object detection into objectness detection and fine-grained classification, which speeds up inference and training with only marginal mAP drop. It is trained on ImageNet Classification data with bounding boxes and obtains 34.9% mAP on ImageNet Detection Dataset (37.8% with SNIPER).

With generalized objectness detection, we demonstrate that it is possible to learn a universal objectness detector. With the universal objectness detector of R-FCN-3k, we can obtain a detector on anything in seconds by learning only the classification layer.

[R-FCN-3k](https://arxiv.org/abs/1712.01802) is described in the following paper:

```
R-FCN-3000 at 30fps: Decoupling Detection and Classification
Bharat Singh*, Hengduo Li*, Abhishek Sharma, Larry S. Davis (* denotes equal contribution)
CVPR, 2018.
```


## Demo

With the trained universal objectness detector, you can obtain a new detector simply by training a light linear classifier in seconds!

1. Please download trained R-FCN-3k model [[GoogleDrive]](https://drive.google.com/file/d/10QOmpklDcY2eO-Lfc3IjY0b9XcTEiYY8/view?usp=sharing)[[BaiduYun]](https://pan.baidu.com/s/1JdxL6B1K3HD_8DjWcrAOwQ) and put them into
```
SNIPER/output/chips_resnet101_3k/res101_mx_3k/fall11_whole/
```

2. Download images [[GoogleDrive]](https://drive.google.com/open?id=1fYsCF6q-bctZMNrLPQkNHJEVxL5LpnYM)[[BaiduYun]](https://pan.baidu.com/s/13HmwE8NdksogxTJb_gFwiw) for new classes and put it in `demo/`.

3. Run `python demo.py` to extract features and train the classifier on new classes. Visualization of detection results on evaluation images are saved in `vis_result`.

4. You can use your own data to train the classifier and obtain a detector. Simply put image folders under `demo/images/` (like demo/images/cat/xxx.jpg) and run `python demo.py`. You may need to change train eval split strategy and hyper-parameters based on your own data and purpose.

## Training

1. Please download [ImageNet Full Fall 2011 Release](http://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47/tech&dllist=1) and [ILSVRC2013_DET](http://www.image-net.org/challenges/LSVRC/2013/download-images-rpa) validation images, together with the [bounding boxes](http://image-net.org/download-bboxes).

2. Download the modified ILSVRC2014_devkit [[GoogleDrive]](https://drive.google.com/open?id=1hEG-GmMrvp--hWRU41RMBLB3gL-IdXs9)[[BaiduYun]](https://pan.baidu.com/s/1wEku413rss02YQ_R39gNGA) which contains essential files for training and evaluation. Please make them look like this:

```
    data
    |__ imagenet
        |__ fall11_whole
            |__ n04233124
                |__ xxx.JPEG
                    ...
                ...
        |__ fall11_whole_bbox
            |__ n04233124
                |__ xxx.xml
                    ...
                ...
        |__ ILSVRC2013_DET_val
        |__ ILSVRC2013_DET_val_bbox
        |__ ILSVRC2014_devkit
            |__ data
                |__ 3kcls_1C_words.txt
                |__ 3kcls_cluster_interval1.txt
                |__ 3kcls_index.txt
                |__ wnid_name_dict.txt
                |__ 3kcls_cluster_result1.txt
                |__ meta_det.mat
                |__ det_lists
                    |__ val.txt
                        ...
            |__ evaluation
                |__ eval_det_3k_1C.m
                    ...
                |__ 3k_1C_pred
                    |__ 3k_1C_matching.txt
                |__ cache
```


3. Run the following script downloads and extract the pre-trained models into the default path (```data/pretrained_model```):
```
bash download_imgnet_models.sh
```

4. To train R-FCN-3k, use the following command:
```
python main_train.py
```

## Evaluation

1. Please download trained R-FCN-3k model [[GoogleDrive]](https://drive.google.com/file/d/10QOmpklDcY2eO-Lfc3IjY0b9XcTEiYY8/view?usp=sharing)[[BaiduYun]](https://pan.baidu.com/s/1JdxL6B1K3HD_8DjWcrAOwQ) and put them into
```
/home/ubuntu/3ksniper/SNIPER/output/chips_resnet101_3k/res101_mx_3k/fall11_whole/
```

2. To evaluate trained model, use the following command:
```
python main_test.py
```

### Citing
```
@article{singh2017r,
  title={R-FCN-3000 at 30fps: Decoupling Detection and Classification},
  author={Singh, Bharat and Li, Hengduo and Sharma, Abhishek and Davis, Larry S},
  journal={CVPR},
  year={2018}
}
```
