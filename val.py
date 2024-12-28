import argparse
import numpy as np
from ultralytics import YOLO

def main(args):
    onnxpath=args.weights
    metrics=YOLO(onnxpath).val(data=args.data,imgsz=args.imgsz)
    if hasattr(metrics,"coco"):
        head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl"
        data=metrics.coco
    else:
        data=np.array([metrics.box.map50,metrics.box.map])
        head="mAP50,mAP"
    np.savetxt(args.save_metrics, np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s-obb.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", type=int, default=1024, help="")
    parser.add_argument("--save-metrics", type=str, default="dota_fp32_yolov8s.txt", help="")
    args = parser.parse_args()
    main(args)