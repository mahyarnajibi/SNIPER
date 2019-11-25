import sys
import logging
import os
import matplotlib
matplotlib.use('Agg')

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'
logging.basicConfig(level=logging.INFO)
sys.path.insert(0,'lib')
sys.path.insert(0,'SNIPER-mxnet/python')
