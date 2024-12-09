"""
_summary_
这是一个用于将ReID pytorch模型转换为onnx模型的脚本。
"""
import torch
from model.make_model_clipreid import make_model
from config import cfg

class OnnxTransformer:
    def __init__(self,config_file,device,onnx_out_path=None) -> None:
        cfg.merge_from_file(config_file)
        self.cfg = cfg
        self.cfg.freeze()
        self.device = device
        self.model = None
        self.onnx_out_path = onnx_out_path

    def getModel(self):
        # ./inference.py partial content
        # However, if you want to export different model,please edit code here.
        model = make_model(self.cfg,1041,0,0)
        model.eval()
        model.to(self.device)
        model.load_param(self.cfg.TEST.WEIGHT)
        self.model = model

    def transformModel(self):
        assert self.model is not None, "Please get pytorch model before transform."
        dummy_input = torch.randn(1, 3, *self.cfg.INPUT.SIZE_TRAIN).cpu()
        self.model.eval().cpu()
        torch.onnx.export(self.model, dummy_input, 
                          self.onnx_out_path if self.onnx_out_path is not None else "model.onnx",
                          export_params=True, opset_version=14)

# example
transformer = OnnxTransformer('configs/person/vit_clipreid.yml',
                              'cuda:2')
transformer.getModel()
transformer.transformModel()