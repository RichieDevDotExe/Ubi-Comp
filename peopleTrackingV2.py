from ultralytics import YOLO
import os
import time
from pathlib import Path
from ultralytics.utils.plotting import save_one_box

def scanRoom():
    model = YOLO('yolov8n.pt')
    results = model.track(source="0",show=True, stream=True,classes=0)
    timecheck = 0

    startTime = time.time()
    count = 0
    for result in results:
        names = model.names
        peopleDetected = []
        resArray = result.numpy()
        print("\n=======================Results Process======================\n")
        #print(resArray)
        #result.save_crop("IDs")
        if(type(resArray.boxes) != type(None)):
            boxes = resArray.boxes
            
            if(type(boxes.id) != type(None)):
                for i in range(0,len(boxes.id)):
                    print("boxes- " + str(boxes.id)+ str(boxes.id.shape))
                    folder = 'IDs'
                    IDval = str(boxes.id[i])
                    path = folder+"/"+IDval
                    if not os.path.exists(path):
                        print("Path no")
                        os.mkdir(path)
                        save_one_box(result.boxes.xyxy[i],resArray.orig_img.copy(),file=Path(path) / f"{Path(IDval)}.jpg",BGR=True)
                    else:
                        save_one_box(result.boxes.xyxy[i],resArray.orig_img.copy(),file=Path(path) / f"{Path(IDval)}.jpg",BGR=True)
                    print("person " + str(boxes.id[i].item())+ " is in frame")
                    
            
            for p in boxes.cls:
                peopleDetected.append(int(p))

            #function to set temperature with peopleDetected as input

            timecheck = time.time()-startTime 
        print("\n=======================IDs detected======================\n")
        print(peopleDetected)
    return peopleDetected

def combineIDs():
    #hello
    print("combine")

def train():
    model = YOLO('yolov8n.pt', device='gpu')

    results = model.train(data='Ubi-Comp-Me-2//data.yaml',epochs = 100, imgsz=640, device=0)


#scanRoom()

