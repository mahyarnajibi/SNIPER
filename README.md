# SNIPER: Efficient Multi-Scale Training

<p align="center">
<img src="http://legacydirs.umiacs.umd.edu/~najibi/github_readme_files/sniper.gif" />
 </p>

SNIPER is an efficient multi-scale training approach for instance-level recognition tasks like object detection and instance-level segmentation. 
Instead of processing all pixels in an image pyramid, SNIPER selectively processes context regions around the ground-truth objects (a.k.a *chips*).
This significantly speeds up multi-scale training as it operates on low-resolution chips. 
Due to its memory efficient design, SNIPER can benefit from *Batch Normalization* during training and it makes larger batch-sizes possible for instance-level recognition tasks on a single GPU. Hence, we do not need to synchronize batch-normalization statistics across GPUs and we can train object detectors similar to the way we do image classification!

[SNIPER](https://arxiv.org/abs/1805.09300) is initially described in the following paper published at NIPS 2018:

<div class="highlight highlight-html"><pre>
<b>SNIPER: Efficient Multi-Scale Training
<a href=https://github.com/bharatsingh430>Bharat Singh*</a>, <a href=https://github.com/mahyarnajibi>Mahyar Najibi*</a>, and Larry S. Davis (* denotes equal contribution)</b>
NIPS, 2018.
</pre></div>

### Features
1. Train with a batch size of 160 images with a ResNet-101 backbone on 8 V100 GPUs
2. NO PYTHON LAYERS (Every layer is optimized for large batch sizes in CUDA/C++)
3. HALF PRECISION TRAINING with no loss in accuracy
4. 5 Images/second during inference on a single V100 GPU, 47.8/68.2 on COCO using ResNet-101 and without training on segmentation masks
5. Use the lightweight MobileNetV2 model trained with SNIPER to get 34.3/54.5 on COCO without training on segmentation masks
6. The R-FCN-3K branch is also powered by SNIPER. Now 21% better than YOLO-9000 on ImageNetDet. This branch also supports on-the-fly training (in seconds) with very few samples (no bounding boxes needed!)
7. Train on OpenImagesV4 (14x bigger than COCO) with ResNet-101 in 3 days on a p3.x16.large AWS instance! 

### Results
Here are the *COCO* results for SNIPER trained using this repository. The models are trained on the *trainval* set (using only the bounding box annotations) and evaluated on the *test-dev* set.

|                                 | <sub>network architecture</sub> | <sub>pre-trained dataset</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|
| <sub>SNIPER </sub>           | <sub>ResNet-101</sub> | <sub>ImageNet</sub> | 46.5 | 67.5    |   52.2  | 30.0  | 49.4  | 58.4  | 
| <sub>SNIPER</sub> |<sub>ResNet-101</sub>  | <sub>OpenImagesV4</sub> | 47.8 |  68.2   | 53.6   | 31.5  | 50.4  | 59.8  |
| <sub>SNIPER</sub> | <sub>MobileNetV2</sub> | <sub>ImageNet</sub> | 34.3 |  54.4   | 37.9   | 18.5  | 36.9  | 46.4  |

You can download the OpenImages pre-trained model by running ```bash scripts/download_pretrained_models.sh```. The SNIPER detectors based on both *ResNet-101* and *MobileNetV2* can be downloaded by running ```bash scripts/download_sniper_detector.sh```.

### License
SNIPER is released under Apache license. See LICENSE for details.

### Citing
```
@article{sniper2018,
  title={{SNIPER}: Efficient Multi-Scale Training},
  author={Singh, Bharat and Najibi, Mahyar and Davis, Larry S},
  journal={NIPS},
  year={2018}
}
@article{analysissnip2017,
  title={An analysis of scale invariance in object detection-snip},
  author={Singh, Bharat and Davis, Larry S},
  journal={CVPR},
  year={2018}
}
```

### Contents
1. [Installation](#install)
2. [Running the demo](#demo)
3. [Training a model with SNIPER](#training)
4. [Evaluting a trained model](#evaluating)
5. [Other methods and branches in this repo (SSH Face Detector, R-FCN-3K, open-images)](#others)

<a name="install"> </a>
### Installation
1. Clone the repository:
```
git clone --recursive https://github.com/mahyarnajibi/SNIPER.git
```

2. Compile the provided MXNet fork in the repository. 

You need to install *CUDA*, *CuDNN*, *OpenCV*, and *OpenBLAS*. These libraries are set to be used by default in the provided ```config.mk``` file in the ```SNIPER-mxnet``` repository. You can use the ```make``` command to build the MXNet library:
```
cd SNIPER-mxnet
make -j [NUM_OF_PROCESS] USE_CUDA_PATH=[PATH_TO_THE_CUDA_FOLDER]
```

If you plan to train models on multiple GPUs, it is optional but recommended to install *NCCL* and build MXNet with the *NCCL* support as instructed below:
```
make -j [NUM_OF_PROCESS] USE_CUDA_PATH=[PATH_TO_THE_CUDA_FOLDER] USE_NCCL=1 
```
In this case, you may also need to set the ```USE_NCCL_PATH``` variable in the above command to point to your *NCCL* installation path.

If you need more information on how to compile MXNet please see [*here*](https://mxnet.incubator.apache.org/install/build_from_source.html).

3. Compile the C++ files in the lib directory. The following script compiles them all:
```
bash scripts/compile.sh
```

4. Install the required python packages:
```
pip install -r requirements.txt
```

<a name="demo"> </a>
### Running the demo

<p align="center">
<img src="http://legacydirs.umiacs.umd.edu/~najibi/github_readme_files/sniper_detections.jpg" width="700px"/>
</p>

For running the demo, you need to download the provided SNIPER model. The following script downloads the SNIPER model and extracts it into the default location:
```
bash download_sniper_detector.sh
```
After downloading the model, the following command would run the SNIPER detector with the default configs on the provided sample image:
```
python demo.py
```
If everything goes well, the sample detections would be saved as ```data/demo/demo_detections.jpg```.

You can also run the detector on an arbitrary image by providing its path to the script:
```
python demo.py --im_path [PATH to the image]
```
However, if you plan to run the detector on multiple images, please consider using the provided multi-process and multi-batch ```main_test``` module. 

You can also test the provided SNIPER model based on the ```MobileNetV2``` architecture by passing the provided config file as follows:
```
python demo.py --cfg configs/faster/sniper_mobilenetv2_e2e.yml
```

<a name="training"></a>
### Training a model

For training SNIPER on COCO, you first need to download the pre-trained models and configure the dataset as described below.

##### Downloading pre-trained models

Running the following script downloads and extracts the pre-trained models into the default path (```data/pretrained_model```):
```
bash download_pretrained_models.sh
```

##### Configuring the COCO dataset

Please follow the [official COCO dataset website](http://cocodataset.org/#download) to download the dataset. After downloading
the dataset you should have the following directory structure:
 ```
data
   |--datasets
         |--coco
            |--annotations
            |--images
```

##### Training the SNIPER detector

You can train the SNIPER detector with or without negative chip mining as described below.

###### Training with Negative Chip Mining:

Negative chip mining results in a relative improvement in AP (please refer to the [paper](https://arxiv.org/pdf/1805.09300.pdf) for the details). To determine the candidate hard negative regions, SNIPER uses proposals extracted from a proposal network trained for a short training schedule. 

For the COCO dataset, we provide the pre-computed proposals. The following commands download the pre-computed proposals, extracts them into the default path (```data/proposals```), and trains the SNIPER detector with the default parameters:
```
bash download_sniper_neg_props.sh
python main_train.py
```

However, it is also possible to extract the required proposals using this repository (e.g. if you plan to train SNIPER on a new dataset). We provided an all-in-one script which performs all the required steps for training SNIPER with Negative Chip Mining. Running the following script trains a proposal network for a short cycle (i.e. 2 epochs), extract the proposals, and train the SNIPER detector with Negative Chip Mining:
```
bash train_neg_props_and_sniper.sh --cfg [PATH_TO_CFG_FILE]
```

###### Training without Negative Chip Mining:

You can disable the negative chip mining by setting the ```TRAIN.USE_NEG_CHIPS``` to ```False```. This is useful if you plan to try SNIPER on a new dataset or want to shorten the training cycle. In this case, the training can be started by calling the following command:
```
python main_train.py --set TRAIN.USE_NEG_CHIPS False
```

In any case, the default training settings can be overwritten by passing a configuration file (see the ```configs``` folder for example configuration files).
The path to the configuration file can be passed as an argument to the above script using the ```--cfg``` flag.
It is also possible to set individual configuration key-values by passing ```--set``` as the last argument to the module 
followed by the desired key-values (*i.e.* ```--set key1 value1 key2 value2 ...```).

Please note that the default config file has the same settings used to train the released models. 
If you are using a GPU with less amount of memory, please consider reducing the training batch size 
(by setting ```TRAIN.BATCH_IMAGES``` in the config file or passing ```--set TRAIN.BATCH_IMAGES [DISIRED_VALUE]``` as the last argument to the module).
 Also, multi-processing is used to process the data. For smaller amounts of memory, you may need to reduce the number of 
 processes and number of threads according to your system (by setting ```TRAIN.NUM_PROCESS``` and ```TRAIN.NUM_THREAD``` respectively).


<a name="evaluating"></a>
### Evaluating a trained model
*Evaluating the provided SNIPER models*

The repository provides a set of pre-trained SNIPER models which can be downloaded by running the following script:
```
bash download_sniper_detector.sh
```
This script downloads the model weights and extracts them into the expected directory. 
To evaluate these models on coco test-dev with the default configuration, you can run the following script:

```
python main_test.py
```
The default settings can be overwritten by passing the path to a configuration file with the ```--cfg``` flag 
(See the ```configs``` folder for examples). It is also possible to set individual configuration key-values by passing ```--set``` as the last argument to the module 
followed by the desired key-values (*i.e.* ```--set key1 value1 key2 value2 ...```).

Please note that the evaluation is performed in a multi-image per batch and parallel model forward setting. In case of lower GPU memory, please consider reducing the batch size for different scales (by setting ```TEST.BATCH_IMAGES```) or reducing the number of parallel jobs (by setting ```TEST.CONCURRENT_JOBS``` in the config file).

*Evaluating a model trained with this repository*

For evaluating a model trained with this repository, you can run the following script by passing the same configuration file used during the training.
The test settings can be set by updating the ```TEST``` section of the configuration file (See the ```configs``` folder for examples).
```
python main_test.py --cfg [PATH TO THE CONFIG FILE USED FOR TRAINING]
```
By default, this would produce a ```json``` file containing the detections on the ```test-dev``` which can be zipped and uploaded to the COCO evaluation server.

<a name="others"></a>
## Branches in this repo (SSH Face Detector, R-FCN-3K, Soft Sampling)
#### R-FCN-3K
This repo also contains the [R-FCN-3k](https://arxiv.org/abs/1712.01802) detector. 
<p align="center">
<img src="http://legacydirs.umiacs.umd.edu/~najibi/github_readme_files/rfcn_3k.jpg" width="600px"/>
</p>

Please switch to the [R-FCN-3k](https://github.com/mahyarnajibi/SNIPER/tree/cvpr3k) branch for specific instructions.

#### OpenImagesV4 with Soft Sampling
This repo also contains modules to train on the [open-images dataset](https://storage.googleapis.com/openimages/web/index.html). 
Please switch to the [openimages2](https://github.com/mahyarnajibi/SNIPER/tree/openimages2) branch for specific instructions. The detector on OpenImagesV4 was trained with [Soft Sampling](https://arxiv.org/abs/1806.06986).

<p align="center">
<img src="http://www.cs.umd.edu/~bharat/ss.jpg" width="650px"/>
</p>

#### SSH Face Detector
The [SSH](https://arxiv.org/abs/1708.03979) face detector would be added to this repository soon. In the meanwhile, you can use the code available at the original [SSH repository](https://github.com/mahyarnajibi/SSH).

<p align="center">
<img src="http://legacydirs.umiacs.umd.edu/~najibi/github_readme_files/ssh_detections.jpg" width="650px"/>
</p>
