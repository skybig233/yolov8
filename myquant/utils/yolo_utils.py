import os
import numpy as np
import torch
from ultralytics import YOLO

def load_calib_dataset(calib_path,batchsize,imgsz):
    import torchvision.transforms as transforms
    from PIL import Image
    # create dataloader
    imgs = []
    trans = transforms.Compose([
        transforms.Resize([imgsz, imgsz]),  # [h,w]
        transforms.ToTensor(),
    ])
    img_path=os.path.join(calib_path,"images")
    label_path=os.path.join(calib_path,"labels")
    for file in os.listdir(path=img_path):
        path = os.path.join(img_path, file)
        img = Image.open(path).convert('RGB')
        img = trans(img)
        imgs.append(img) # img is 0 - 1

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=imgs, batch_size=batchsize)
    return dataloader


def load_yolov5_model(model_path):
    # 确保设备有效
    device = torch.device('cuda',0)
    # 加载模型
    model = torch.hub.load('../yolov5', 'custom', model_path, source='local')
    # 将模型转移到设备上
    model.to(device)
    # 设置为预测模式
    model.eval()
    return model



def load_yolov8_model(model_path):
    return YOLO(model_path).model


def load_model(model_path:str):
    if "yolov5" in model_path:
        return load_yolov5_model(model_path)
    elif "yolov8" in model_path:
        if model_path.endswith(".pt"):
            print("不使用yolov8导出的onnx模型可能会报错")
        return load_yolov8_model(model_path)
    else:
        raise Exception
    

def val_yolov5_model(onnx_path,save_metrics_txt,cfg,imgsz):
    # os.system(f"python /home/jzs/cv/yolov5/val.py --weights {onnx_path} --data coco.yaml --device 0")
    os.system(f"python /home/jzs/cv/yolov5/val.py --weights {onnx_path} --data {cfg} --device 0 --imgsz {imgsz} --save-metrics {save_metrics_txt}")

def val_yolov8_model(onnx_path,save_metrics_txt,cfg,imgsz):
    os.system(f"python /home/jzs/cv/ultralytics/val.py --weights {onnx_path} --data {cfg} --imgsz {imgsz} --save-metrics {save_metrics_txt}")

def val_model(onnx_path,save_metrics_txt,cfg,imgsz):
    if "yolov5" in onnx_path:
        val_yolov5_model(onnx_path,save_metrics_txt,cfg,imgsz)
    elif "yolov8" in onnx_path:
        val_yolov8_model(onnx_path,save_metrics_txt,cfg,imgsz)
    else:
        raise Exception


if __name__=="__main__":
    onnx="/home/jzs/cv/ppq/trt_output/yolov8s_ppqexport.onnx"
    txt="test.txt"
    cfg="/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/coco.yaml"
    val_yolov8_model(onnx,txt,cfg,640)