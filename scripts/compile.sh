#!/bin/bash
echo 'Compiling NMS module...'
(cd lib/nms; python setup_linux.py build_ext --inplace)
echo 'Compiling bbox module...'
(cd lib/bbox; python setup_linux.py build_ext --inplace)
#echo 'Compiling chips module...'
#(cd lib/chips; python setup.py)
echo 'Compiling coco api...'
(cd lib/dataset/pycocotools; python setup_linux.py build_ext --inplace)
echo 'All Done!'
