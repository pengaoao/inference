"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt
import numpy as np
import backend
import torch


class BackendOnnxruntime(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, use_gpu=False, mini_batch_size=1):
        super(BackendOnnxruntime, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

    def load(self, model_path, inputs=None, outputs=None):
        
        inputs = None
        outputs = None
        print("onnx load", model_path, inputs, outputs)
        """Load model and find input/outputs from the model file."""
        opt = rt.SessionOptions()
        # enable level 3 optimizations
        # FIXME: enable below once onnxruntime 0.5 is released
        # opt.set_graph_optimization_level(3)
        self.sess = rt.InferenceSession(model_path, opt)
        
        # get input and output names
        self.inputs = [meta.name for meta in self.sess.get_inputs()]
        self.outputs = [meta.name for meta in self.sess.get_outputs()]
        
        self.overloads = [meta.name for meta in self.sess.get_overridable_initializers()] 
        self.meta = self.sess.get_modelmeta() 
        #self.prov_option = self.sess.get_provider_options()
        
        print("inputs", self.inputs)
        print("outputs", self.outputs)
        print("overloads", self.overloads)
        print("meta", self.meta)
        #print("prov_option", self.prov_option)
        
        #self.outputs = ["predict"]
        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
       
#        print("onnx predict batch", batch_dense_X, '\n\n', batch_lS_o, '\n\n', batch_lS_i)
        """Run the prediction."""
        print("batch_lS_i.shape", batch_lS_i.shape)
        
        dict_inputs = {}
        # dict_inputs[self.inputs[0]] = batch_dense_X.numpy().astype(np.float32)
        # dict_inputs[self.inputs[1]] = batch_lS_o.numpy().astype(np.int64)
        # dict_inputs[self.inputs[2]] = batch_lS_i.numpy().astype(np.int64)

        ind = 0
        for i in self.inputs:
            if ind == 0:               # dense features
                dict_inputs[i] = batch_dense_X.numpy().astype(np.float32)
            elif ind == 1:             # offsets
                dict_inputs[i] = batch_lS_o.numpy().astype(np.int64)
            else:                      # indices
                dict_inputs[i] = batch_lS_i[ind-2].numpy().astype(np.int64)

            ind = ind + 1

        prediction = self.sess.run(output_names=self.outputs, input_feed=dict_inputs)
       # print("prediction", prediction)
        
        return torch.tensor(prediction, requires_grad=False).view(-1,1)
