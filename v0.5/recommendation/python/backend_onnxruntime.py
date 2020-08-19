"""
onnxruntime backend (https://github.com/microsoft/onnxruntime)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import onnxruntime as rt

import backend


class BackendOnnxruntime(backend.Backend):
    def __init__(self, m_spa, ln_emb, ln_bot, ln_top, use_gpu=False, mini_batch_size=1):
        super(BackendOnnxruntime, self).__init__()

    def version(self):
        return rt.__version__

    def name(self):
        """Name of the runtime."""
        return "onnxruntime"

#    def image_format(self):
#        """image_format. For onnx it is always NCHW."""
#        return "NCHW"

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
        if inputs is None:
            self.inputs = [meta.name for meta in self.sess.get_inputs()]
        else:
            self.inputs = inputs
        
        if outputs is None:
            self.outputs = [meta.name for meta in self.sess.get_outputs()]
        else:
            self.outputs = outputs
        
        print("inputs", self.inputs)
        print("outputs", self.outputs)
        #self.outputs = ["predict"]
        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
        print("onnx predict")
        """Run the prediction."""
        return self.sess.run(output_names=self.outputs, input_feed=[{self.inputs[0]:batch_dense_X.numpy()}, {self.inputs[1]:batch_lS_o.numpy()}, {self.inputs[2]:batch_lS_i.numpy()}])
        