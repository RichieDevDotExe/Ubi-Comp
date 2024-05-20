from ultralytics import YOLO
import os
import time
from pathlib import Path
from ultralytics.utils.plotting import save_one_box
#import torch 

#device = torch.device("cpu")

def main():
    model = YOLO('best.pt')
    results = model.track(source="0",show=True, stream=True,conf = 0.7)
    for result in results:
        print(trainedScan(result))

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

# Dictionary stores preferred temperature in celsius
tempPreferences = {
    0: 19,
    1: 16,
    2: 22,
    3: 21,
    4: 17
}
# Baseline temperature, for if nobody is in the room
baselineTemp = 15
# function to decide a target temperature
def getTargetTemp(inRoom):
    # If nobody's in the room, keep it at baseline temperature
    targetTemp = baselineTemp

    # Otherwise, if somebody's in there, our target is the preference average
    if inRoom:
        targetTemp = 0
        for person in inRoom:
            # If the person is in the dictionary then use their preference 
            if person in tempPreferences:
                targetTemp += tempPreferences[person]
            # Otherwise use baseline temperature
            else:
                targetTemp += baselineTemp
        targetTemp /= len(inRoom)
    
    return targetTemp

def scanRoomL():
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
                    peopleDetected.append(int(boxes.id[i]))
                    
            
            #for p in boxes.cls:
            #    peopleDetected.append(int(p))

            #function to set temperature with peopleDetected as input

            timecheck = time.time()-startTime 
        print("\n======IDs detected======\n")
        print(peopleDetected)

        # Target temperature decided here
        targetTemp = getTargetTemp(peopleDetected)
        print("Target temperature: "+str(targetTemp))
    #return peopleDetected

def combineIDs():
    #hello
    print("combine")

def train():
    model = YOLO('yolov8n.pt')
    #model.to(device)

    results = model.train(data='Ubi-Comp-Me-2//data.yaml',epochs = 100, imgsz=640, device=0)

def trainedScan(result):
    peopleDetected = []
    resArray = result.numpy()
    if(type(resArray.boxes) != type(None)):
        boxes = resArray.boxes

        for p in boxes.cls:
            peopleDetected.append(int(p))
    
    return peopleDetected

# #loosly based off sample code found in person_reid.py in openCV library
# def collect_images(pathToImgs):
#     img_list = []
#     with os.scandir(path=pathToImgs) as entries:
#         for entry in entries:
#             if entry.is_file():
#                 img = cv.imread(entry.path)
#                 if img is None:
#                     continue
#                 img_list.append(img)
#     return np.array(img_list)

# def scanIDFolders(pathToIDFolders):
#     idImages = []
#     with os.scandir(path=pathToIDFolders) as Folders:
#         for folder in Folders:
#             if folder.is_dir():
#                 idList = collect_images(folder.path)
#                 idImages.append(idList)
#             #print("foldername "+ str(folder.name))
#             #print("folder legnth "+ str(len(idList)))
#     return np.array(idImages)



def combineIDs():
    #hello
    print("combine")

#get model from https://drive.google.com/drive/folders/1wFGcuolSzX3_PqNKb4BAV3DNac7tYpc2     
img_feat, img_names = REID.extract_feature(img_dir="IDs//1.0",model_path="youtu_reid_baseline_lite.onnx",batch_size=32, resize_h=256,resize_w=128,backend=cv.dnn.DNN_BACKEND_OPENCV,target=cv.dnn.DNN_TARGET_CPU)
print(img_feat)
print("=========")
print(img_names)

#scanRoom()

#trainedScan()

main()
