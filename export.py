import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from myquant.lunwen.drawfig import hook_fn
from myquant.lunwen.lunwen import AWQObserver, SmoothConv2d, awq_calib_model, awq_quantize_model
from myquant.utils.yolo_utils import load_calib_dataset
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv, MSQConv
from ultralytics.utils import LOGGER
from torch import nn

def save_scale_json(scale):
    # 先用普通的json dump
    json_str = json.dumps(scale, indent=4)
    # 将数组中的逗号+空格替换为逗号+换行
    json_str = json_str.replace('], "', '],\n    "')  # 处理数组结束
    json_str = json_str.replace(', ', ',\n        ')  # 处理数组内元素
    return json_str



def main(args):
    modelpath=args.weights
    model=YOLO(modelpath)
    # model.val(data="coco128.yaml")
    model.fuse()
    with open(args.scale, "r") as f:
        migscale_data = json.load(f)

    def apply_migscale_to_conv(model, migscale_data, prefix=""):
        """
        Apply migscale to nn.Conv2d modules in the model.
        
        Args:
            model (nn.Module): The model to process.
            migscale_data (dict): Dictionary with module names as keys and migscale values as values.
            prefix (str): Current module name prefix for nested modules.
        """
        for name, module in model.named_children():
            # Build full module name (e.g., "model.model.0.conv")
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, nn.Conv2d):
                # Check if full_name exists in migscale_data
                if full_name not in migscale_data:
                    raise KeyError(f"Module '{full_name}' not found in migscale data.")

                # Retrieve migscale for the current module
                migscale = torch.tensor(migscale_data[full_name]).view(1,-1,1,1).to(module.weight.device)

                # Adjust the module's weight
                module.weight.data *= migscale

                # Wrap the forward method to adjust input
                original_forward = module.forward

                def new_forward(x, migscale=migscale, original_forward=original_forward):
                    return original_forward(x / migscale)

                # Replace the forward method
                module.forward = new_forward
            else:
                # Recursively process nested modules
                apply_migscale_to_conv(module, migscale_data, prefix=full_name)
    
    # apply_migscale_to_conv(model, migscale_data)
    # model.export(format="openvino",opset=13,int8=True,data="coco.yaml")
    model.export(format="openvino",opset=13,data="coco.yaml")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", type=str, default="myquant/lunwen/runs/val/a_rebuttal_coco_yolov8s_W8A8_pertensor_wsym_aasym_my/calibed_quantized_migrated.pt", help="initial weights path")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="initial weights path")
    # parser.add_argument("--weights", type=str, default="yolov8s-obb.pt", help="initial weights path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/VOC.yaml", help="dataset.yaml path")
    parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/coco.yaml", help="dataset.yaml path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml", help="dataset.yaml path")

    # parser.add_argument("--imgsz", type=int, default=512, help="")
    parser.add_argument("--imgsz", type=int, default=640, help="")
    # parser.add_argument("--imgsz", type=int, default=1024, help="")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--scale",type=str,default="myquant/lunwen/runs/val/a_rebuttal_coco_yolov8s_W8A8_pertensor_wsym_aasym_my/scale.json")
    args = parser.parse_args()
    main(args)