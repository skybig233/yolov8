import os
code="/home/jzs/cv/ultralytics/myquant/lunwen/val_lunwen.py"

weights=[
    "/home/jzs/cv/ultralytics/runs/detect/voc_yolov8n/weights/best.pt",
    "/home/jzs/cv/ultralytics/runs/detect/voc_yolov8s/weights/best.pt"
]
# weights=[
#     "yolov8n-obb.pt",
#     "yolov8s-obb.pt"
# ]
q=[
    ("per_tensor","per_tensor_percentile"),
    ("per_tensor","per_tensor_asym")
]
q_name=[
    ("wsym","apercentile9999"),
    ("wsym","aasym")
]
project="/home/jzs/cv/ultralytics/myquant/lunwen/runs/val"

for w in weights:
    for i in range(len(q)):
        wq,aq=q[i][0],q[i][1]
        wqname,aqname=q_name[i][0],q_name[i][1]
        weight=w
        data="dota1"
        # data_cfg=f"/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/{data}.yaml"
        data_cfg=f"/home/jzs/cv/ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml"

        calib=f"/home/jzs/cv/ppq/{data}imgs"
        # voc
        # name=data+"_"+w.split("/")[-3].split("_")[-1]+f'_W8A8_pertensor_{wqname}_{aqname}_awq'
        # coco
        name=data+"_"+w.split("-")[0]+f'_W8A8_pertensor_{wqname}_{aqname}_awq'
        # imgsz=512
        # imgsz=640
        imgsz=1024
        mc=f"{project}/{name}"
        cmd=f"python {code} --weights {weight} --imgsz {imgsz} --data {data_cfg} --calib {calib} --name {name} --save-metrics {mc}/mc.txt --wq {wq} --aq {aq} --project {project}" 
        # print(cmd)
        os.system(cmd)