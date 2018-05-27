import mxnet as mx
import numpy as np
from nms import nms
import random
DEBUG = True
# newanchor_boxes2 = mx.sym.Custom(op_type='Det2proposal', name='det2proposal', det=det, anchor=anchor_boxes2)
class det2proposal(mx.operator.CustomOp):
    def __init__(self):
        super(det2proposal, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        det = in_data[0]
        anchor = in_data[1].asnumpy()
        anchor_number = anchor.shape[1]
        det = mx.nd.reshape(det, shape=(-3, 6))
        classes = mx.nd.slice(det, begin=(0,0), end=(anchor_number, 1))
        prob = mx.nd.slice(det, begin=(0,1), end=(anchor_number, 2))
        classes = classes.asnumpy()
        prob = prob.asnumpy()
        positive = list(np.where(classes > -1)[0])
        negtive = list(np.where(classes == -1)[0])
        keep_negtive = negtive[len(negtive)-0*len(positive):len(negtive)]
        # keep anchor include positive and negtive
        keep = positive + keep_negtive
        anchor = mx.nd.slice(det, begin=(0,2), end=(anchor_number, None))
        anchor = anchor.asnumpy()
        new_anchor = np.zeros((anchor_number, 4))
        new_anchor[keep] = anchor[keep]
        anchor = mx.nd.array(new_anchor)
        anchor = mx.nd.reshape(anchor, shape=(-4, 1, -1, -2))
        self.assign(out_data[0], req[0], anchor)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('Det2proposal')
class det2proposalProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(det2proposalProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['det', 'anchor']

    def list_outputs(self):
        return ['proposal']

    def infer_shape(self, in_shape):
        # arg_shapes, out_shapes, aux_shapes
        detshape = in_shape[0]
        anchorshape = in_shape[1]
        out_shapes = in_shape[1]
        return [detshape, anchorshape], [out_shapes], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return det2proposal()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

