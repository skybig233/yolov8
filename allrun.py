import numpy as np
from ultralytics import YOLO

# yolov5n="python train.py --weight yolov5n.pt --data /home/jzs/cv/yolov5/data/VOC.yaml --cfg /home/jzs/cv/yolov5/models/yolov5n.yaml --epoch 200 --batch-size 64 --imgsz 500 --name voc_yolov5n"
# yolov5s="python train.py --weight yolov5s.pt --data /home/jzs/cv/yolov5/data/VOC.yaml --cfg /home/jzs/cv/yolov5/models/yolov5s.yaml --epoch 200 --batch-size 64 --imgsz 500 --name voc_yolov5s"
# yolov5m="python train.py --weight yolov5m.pt --data /home/jzs/cv/yolov5/data/VOC.yaml --cfg /home/jzs/cv/yolov5/models/yolov5m.yaml --epoch 200 --batch-size 32 --imgsz 500 --name voc_yolov5m"
# yolov5l="python train.py --weight yolov5l.pt --data /home/jzs/cv/yolov5/data/VOC.yaml --cfg /home/jzs/cv/yolov5/models/yolov5l.yaml --epoch 200 --batch-size 32 --imgsz 500 --name voc_yolov5l"
# yolov5x="python train.py --weight yolov5x.pt --data /home/jzs/cv/yolov5/data/VOC.yaml --cfg /home/jzs/cv/yolov5/models/yolov5x.yaml --epoch 200 --batch-size 32 --imgsz 500 --name voc_yolov5x"



# model_size=["n","s","m","l","x"]
model_size=["l","x"]
# batch_size=[64,64,32,32,16]
batch_size=[32,16]
for i in range(len(model_size)):
    # Load a model
    model = YOLO(f"yolov8{model_size[i]}.yaml")  # build a new model from scratch
    model = YOLO(f"yolov8{model_size[i]}.pt")  # load a pretrained model (recommended for training)    

    # Val
    # metrics=model.val(data="coco.yaml",name=f"coco_fp32_yolov8{i}")
    # if hasattr(metrics,"coco"):
    #     head="AP,AP50,AP75,APs,APm,APl,AR1,AR10,AR100,ARs,ARm,ARl"
    #     data=metrics.coco
    # else:
    #     data=np.array([metrics.box.map50,metrics.box.map])
    #     head="mAP50,mAP"
    # np.savetxt(f"coco_fp32_yolov8{i}.txt", np.reshape(data,(1,-1)), fmt='%.4f', delimiter=',',header=head)

    # Use the model
    # train the model
    model.train(data="/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/VOC.yaml", epochs=200,batch=batch_size[i],imgsz=500,name=f"voc_yolov8{model_size[i]}")
      # resume train the model
    # model.train(data="/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/VOC.yaml", epochs=200,batch=batch_size[i],imgsz=500,name=f"voc_yolov8{model_size[i]}",resume=True,device='0')
    # metrics = model.val()  # evaluate model performance on the validation set
    # path = model.export(format="onnx")  # export the model to ONNX format
