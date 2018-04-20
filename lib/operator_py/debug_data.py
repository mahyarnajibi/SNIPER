import mxnet as mx
import numpy as np
from distutils.util import strtobool
import cPickle

class DebugDataOperator(mx.operator.CustomOp):
    def __init__(self):
        super(DebugDataOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_score_1    = in_data[0].asnumpy()
        cls_score_2    = in_data[1].asnumpy()
        cls_score_3    = in_data[2].asnumpy()
        cls_score_4    = in_data[3].asnumpy()
        cls_score_5    = in_data[4].asnumpy()
        cls_score_6    = in_data[5].asnumpy()                

        with open('d1.pkl', 'wb') as f:
            cPickle.dump(cls_score_1, f, -1)

        with open('d2.pkl', 'wb') as f:
            cPickle.dump(cls_score_2, f, -1)

        with open('d3.pkl', 'wb') as f:
            cPickle.dump(cls_score_3, f, -1)

        with open('d4.pkl', 'wb') as f:
            cPickle.dump(cls_score_4, f, -1)

        with open('d5.pkl', 'wb') as f:
            cPickle.dump(cls_score_5, f, -1)
            
        with open('d6.pkl', 'wb') as f:
            cPickle.dump(cls_score_6, f, -1)
        

        import pdb
        pdb.set_trace()
        self.assign(out_data[0], req[0], in_data[0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('debug_data')
class DebugData(mx.operator.CustomOpProp):
    def __init__(self):
        super(DebugData, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['datai1', 'datai2', 'datai3', 'datai4', 'datai5', 'datai6']

    def list_outputs(self):
        return ['datao']

    def infer_shape(self, in_shape):
        out_shape = in_shape[0]
        return in_shape, [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DebugDataOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
