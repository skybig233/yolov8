import argparse
import numpy as np
from ultralytics import YOLO

def main(args):
    onnxpath=args.weights
    model=YOLO(onnxpath)
    metrics=model.val(data=args.data,imgsz=args.imgsz,project=args.project,name=args.name)
    if hasattr(metrics,"coco"):
        head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl"
        data=metrics.coco
    else:
        data=np.array([metrics.box.map50,metrics.box.map])
        head="mAP50,mAP"
    np.savetxt(args.save_metrics, np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", type=str, default="myquant/lunwen/runs/val/a_rebuttal_coco_yolov8s_W8A8_pertensor_wsym_aasym_my/calibed_quantized_migrated_int8_openvino_model", help="initial weights path")
    parser.add_argument("--weights", type=str, default="yolov8s_openvino_model", help="initial weights path")
    parser.add_argument("--data", type=str, default="../ultralytics/ultralytics/cfg/datasets/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="")
    parser.add_argument("--save-metrics", type=str, default="coco_openvinoint8_yolov8s.txt", help="")
    parser.add_argument("--name", default="a_rebuttal_coco_yolov8s_fp32_speed", help="save to project/name")
    parser.add_argument("--project", default="/home/jzs/cv/ultralytics/runs/val", help="save to project/name")
    args = parser.parse_args()
    main(args)