# Soft Sampling for Robust Object Detection

<p align="center">
<img src="http://www.cs.umd.edu/~bharat/ss.jpg" />
</p>

Soft Sampling re-weights the gradients of RoIs as a function of overlap with positive instances for training robust object detectors with missing labels. This ensures that the uncertain background regions are given a smaller weight compared to the hardnegatives.

[Soft Sampling](https://arxiv.org/abs/1806.06986) is described in the following paper:

<b>Soft Sampling for Robust Object Detection</b> <br>
[Zhe Wu*](https://github.com/Doubaibai), [Navaneeth Bodla*](https://github.com/navaneethbodla), [Bharat Singh*](https://github.com/bharatsingh430), [Mahyar Najibi](https://github.com/mahyarnajibi), Rama Chellappa, Larry S. Davis (* denotes equal contribution) <br>
arXiv preprint arXiv:1806.06986, 2018.
</pre>


### License
Soft-Sampling is released under Apache license. See LICENSE for details.


### Installation
Please refer to the master branch for installation instructions.
<a name="demo"> </a>

### Training
SNIPER makes it possible to train on the OpenImages dataset in 3 days. For training the model with default configs you can use the following script:
```
python main_chips_open.py
```

The model we trained on OpenImagesV4 used Soft-Sampling which is implemented in [multi_proposal_target_layer.cu](https://github.com/mahyarnajibi/SNIPER-mxnet/blob/SNIPER-mxnet-open/src/operator/multi_proposal_target.cu) in the [SNIPER-mxnet-open](https://github.com/mahyarnajibi/SNIPER-mxnet/tree/SNIPER-mxnet-open/) branch.

Please note that while the branch is able to reproduce the results, it is out of sync with the master branch and would be updated soon.
