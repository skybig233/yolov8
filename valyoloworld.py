# # "Custom Inference Prompts"
# from ultralytics import YOLOWorld
# import yaml
# datacfg="/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/lvis128.yaml"
# with open(datacfg, 'r') as file:
#     data = yaml.safe_load(file)
# names = data['names']

# # Initialize a YOLO-World model
# model = YOLOWorld("yolov8s-worldv2.pt")  # or choose yolov8m/l-world.pt


# model.set_classes(list(names.values()))
# # model.export(format="onnx")
# result=model.val(data=datacfg.split("/")[-1])
# print(result)
# Define custom classes
# Execute prediction for specified categories on an image
# model.set_classes(["human"])
# results = model.predict("/home/jzs/cv/datasets/coco128/images/train2017/000000000431.jpg")
# Show results
# results[0].show()

import argparse
import numpy as np
from ultralytics import YOLOWorld
import yaml
def main(args):
    onnxpath=args.weights
    model=YOLOWorld(onnxpath)
    datacfg=args.data
    with open(datacfg, 'r') as file:
        data = yaml.safe_load(file)
    names = data['names']
    # set_classes之后模型才会有clipmodule
    model.set_classes(list(names.values()))

    metrics=model.val(data=args.data,imgsz=args.imgsz)
    if hasattr(metrics,"coco"):
        head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl,speed/preprocess,speed/inference,speed/postprocess"
        data=np.concatenate((metrics.coco,np.array([metrics.speed["preprocess"],metrics.speed["inference"],metrics.speed["postprocess"]])),axis=0)
    else:
        data=np.array([metrics.box.map50,metrics.box.map,metrics.speed["preprocess"],metrics.speed["inference"],metrics.speed["postprocess"]])
        head="mAP50,mAP,speed/preprocess,speed/inference,speed/postprocess"
    args.save_metrics=metrics.save_dir / args.save_metrics
    np.savetxt(args.save_metrics, np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s-worldv2.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/lvis128.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="")
    parser.add_argument("--save-metrics", type=str, default="lvis_test.txt", help="")
    args = parser.parse_args()
    main(args)