import argparse
import os
import numpy as np
from tqdm import tqdm
from myquant.lunwen.lunwen import awq_calib_model, awq_quantize_model
from myquant.utils.yolo_utils import load_calib_dataset
from ultralytics import YOLO

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
    device=model.device
    for batch_i, im in enumerate(pbar_calib):
        if batch_i >= 200 :
            print(batch_i)
            break
        im = im.to(device, non_blocking=True)
        model(im)

    if not (args.wq and args.aq):
        raise Exception("未指定量化方法")
    print(f"weight_quant:{args.wq}\nact_quant{args.aq}")
    model = awq_quantize_model(
        model, 
        weight_quant=args.wq, 
        act_quant=args.aq,
    )
    metrics=model.val(data=args.data,imgsz=args.imgsz,batch=BATCHSIZE,project=args.project,name=args.name)

    args.save_metrics=os.path.join(os.path.join(args.project,args.name),args.save_metrics)
    if hasattr(metrics,"coco"):
        head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl"
        data=metrics.coco
    else:
        data=np.array([metrics.box.map50,metrics.box.map])
        head="mAP50,mAP"
    np.savetxt(args.save_metrics, np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", type=str, default="/home/jzs/cv/ultralytics/runs/detect/voc_yolov8x/weights/best.pt", help="initial weights path")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    # parser.add_argument("--weights", type=str, default="yolov8s-obb.pt", help="initial weights path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/VOC.yaml", help="dataset.yaml path")
    parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/coco.yaml", help="dataset.yaml path")
    # parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml", help="dataset.yaml path")
    # parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/VOCimgs", help="dataset.yaml path")
    parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/cocoimgs", help="dataset.yaml path")
    # parser.add_argument("--calib", type=str, default="/home/jzs/cv/ppq/dota1imgs", help="dataset.yaml path")
    # parser.add_argument("--imgsz", type=int, default=512, help="")
    parser.add_argument("--imgsz", type=int, default=640, help="")
    # parser.add_argument("--imgsz", type=int, default=1024, help="")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--wq", type=str, default="per_tensor", help="per_tensor")
    parser.add_argument("--aq", type=str, default="per_tensor_percentile", help="per_tensor_percentile")
    parser.add_argument("--name", default="coco_yolov8n_W8A8_pertensor_wsym_apercentile9999_my", help="save to project/name")
    parser.add_argument("--project", default="/home/jzs/cv/ultralytics/myquant/lunwen/runs/val", help="save to project/name")
    parser.add_argument("--save-metrics", type=str, default="mc.txt", help="")
    args = parser.parse_args()
    main(args)