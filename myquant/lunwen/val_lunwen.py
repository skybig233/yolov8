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
from ultralytics.utils import LOGGER


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
    model=awq_calib_model(model)

    BATCHSIZE   = args.batch_size
    CALIBRATION_PATH = args.calib                     # 校准数据集
    CALIBRATION = load_calib_dataset(CALIBRATION_PATH,BATCHSIZE,args.imgsz)
    calib_dataloader=CALIBRATION
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
    pbar_calib = tqdm(calib_dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    # device=model.device
    device=torch.device("cuda")

    if not (args.wq and args.aq):
        raise Exception("未指定量化方法")
    print(f"weight_quant:{args.wq}\nact_quant{args.aq}")

        
    # 创建fig目录
    beforemsq_data_dir = 'before_migration_data'
    aftermsq_data_dir="after_migration_data"
    if not os.path.exists(beforemsq_data_dir):
        os.makedirs(beforemsq_data_dir)
    if not os.path.exists(aftermsq_data_dir):
        os.makedirs(aftermsq_data_dir)
    
    # # Calib Model
    # # # 迁移前查看数据分布，注册钩子到每一个卷积层
    # for name, layer in model.named_modules():
    #     if isinstance(layer, AWQObserver):
    #         layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name,beforemsq_data_dir))

    for batch_i, im in enumerate(pbar_calib):
        if batch_i >= 200 :
            print(batch_i)
            break
        im = im.to(device, non_blocking=True)
        model(im)


    # print(model.model.model[1].conv.act_scales)
    # 量化
    opt=args
    if opt.apply_scale:
        with open(opt.apply_scale, 'r') as f:
            apply_scale=json.load(f)
    else:
        apply_scale=opt.apply_scale
        
    save_scale={}
    # Quantize Model
    model=awq_quantize_model(
        root=model,
        model=model, 
        weight_quant=opt.wq, 
        act_quant=opt.aq,
        apply_scale=apply_scale,
        save_scale=save_scale
    )

    # 迁移后查看数据分布
    # for name, layer in model.named_modules():
    #     if isinstance(layer, SmoothConv2d):
    #         layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name,aftermsq_data_dir))

    # Save Model
    # 保存量化后的模型，model是DetectionMultiBackend 
    # model.model是DetectionModel
    # ckpt = {
    #     'epoch': 0,
    #     'best_fitness': 0,
    #     'model': deepcopy(de_parallel(model.model)).half(),
    #     'ema': None,
    #     'updates': None,
    #     'optimizer': None,
    #     'wandb_id': None,
    #     'opt': vars(opt),
    #     'date': datetime.now().isoformat()
    # }

    # save_dir= Path(args.project)


    metrics=model.val(data=args.data,imgsz=args.imgsz,batch=BATCHSIZE,project=args.project,name=args.name)

    if hasattr(metrics,"coco"):
        head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl,speed/preprocess,speed/inference,speed/postprocess"
        data=np.concatenate((metrics.coco,np.array([metrics.speed["preprocess"],metrics.speed["inference"],metrics.speed["postprocess"]])),axis=0)
    else:
        data=np.array([metrics.box.map50,metrics.box.map,metrics.speed["preprocess"],metrics.speed["inference"],metrics.speed["postprocess"]])
        head="mAP50,mAP,speed/preprocess,speed/inference,speed/postprocess"
    args.save_metrics=metrics.save_dir / args.save_metrics
    np.savetxt(args.save_metrics, np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

    if opt.save_scale!='':
        # 如果是迁移学习，存储到opt.save_scale
        opt.save_scale=metrics.save_dir / opt.save_scale
        with open(opt.save_scale, 'w') as f:
            f.write(save_scale_json(save_scale))

    # 保存到weights目录
    save_dir=metrics.save_dir
    save_path = str(save_dir / 'calibed_quantized_migrated.pt')
    LOGGER.info(f'量化模型已保存到: {save_path}')
    model.save(save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", type=str, default="/home/jzs/cv/ultralytics/runs/detect/voc_yolov8s/weights/best.pt", help="initial weights path")
    # parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--weights", type=str, default="yolov8s-obb.pt", help="initial weights path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/VOC.yaml", help="dataset.yaml path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml", help="dataset.yaml path")

    # parser.add_argument("--imgsz", type=int, default=512, help="")
    # parser.add_argument("--imgsz", type=int, default=640, help="")
    parser.add_argument("--imgsz", type=int, default=1024, help="")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--wq", type=str, default="per_tensor", help="per_tensor")
    parser.add_argument("--aq", type=str, default="per_tensor_percentile", help="per_tensor_percentile")
    # parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/VOCimgs", help="dataset.yaml path")
    # parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/cocoimgs", help="dataset.yaml path")
    parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/dota1imgs", help="dataset.yaml path")
    # 保存scale
    parser.add_argument("--save_scale", type=str, default="scale.json", help="save scale file")
    # parser.add_argument("--save_scale", type=str, default="", help="save scale file")
    # 应用scale
    parser.add_argument("--apply_scale", type=str, default="", help="apply scale file")



    parser.add_argument("--name", default="a_rebuttal_dota_yolov8s_W8A8_pertensor_wsym_apercentile_my", help="save to project/name")
    parser.add_argument("--project", default="/home/jzs/cv/ultralytics/myquant/lunwen/runs/val", help="save to project/name")
    parser.add_argument("--save-metrics", type=str, default="mc.txt", help="")
    args = parser.parse_args()
    main(args)