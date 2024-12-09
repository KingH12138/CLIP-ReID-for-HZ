import argparse
from config import cfg
from model.make_model_clipreid import make_model
import torch

def parse_arg():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

def inference(args,cfg,inputs,device):
    """
    args:....
    cfg:从yml文件导入并合并的配置，请注意：cfg是有默认配置的。
    inputs:the tensor like (bs,c,h,w)
    
    outputs: the embedded tensor like (bs,embed_out_dim)
    """
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    model = make_model(cfg,1041,0,0)
    model.eval()
    model.to(device)
    model.load_param(cfg.TEST.WEIGHT)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

# example:
inputs = torch.randn(1,3,256,128)
args = parse_arg()
device = torch.device('cuda:2')
outputs = inference(args,cfg,inputs,device)
print(outputs.shape)