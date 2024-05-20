from ultralytics import YOLO
import os
import time
from pathlib import Path
from ultralytics.utils.plotting import save_one_box
import person_reid as REID
import cv2 as cv
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


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

#get model from https://drive.google.com/drive/folders/1wFGcuolSzX3_PqNKb4BAV3DNac7tYpc2    
def scanIDFolders(pathToIDFolders):
    idFeat = []
    idNames = []
    maxlen = -1
    with os.scandir(path=pathToIDFolders) as Folders:
        for folder in Folders:
            if folder.is_dir():
                img_feat, img_names = REID.extract_feature(img_dir=folder.path,model_path="youtu_reid_baseline_lite.onnx",batch_size=32, resize_h=256,resize_w=128,backend=cv.dnn.DNN_BACKEND_CUDA,target=cv.dnn.DNN_TARGET_CUDA)
                idFeat.append(img_feat)
                idNames.append(folder.name)
    
    #must make arrays the same length for numpy
    padding = np.zeros(len(idFeat[0][0]))
    print(len(idFeat[0][0]))
    for i in range(0,len(idFeat)):
        if(len(idFeat[i]) > maxlen):
            maxlen = len(idFeat[i])
            print("=======cmp==========")
            print(len(idFeat[i]))
            print(maxlen)

    for j in range(0,len(idFeat)):
        paddingsize = maxlen - len(idFeat[j])
        for k in range(paddingsize):
            #print("padding")
            idFeat[j] = np.row_stack((idFeat[j],padding))  


    npIDFeat = np.array(idFeat)
    npIDNames = np.array(idNames)

    return npIDFeat, npIDNames

def visualiseExtraction(idFeat,idNames):
    fig = plt.figure(figsize=(20,15))
    plt.imshow(idFeat[0][0], interpolation='nearest', cmap=plt.cm.Blues)
    plt.savefig("Vis.png",dpi=fig.dpi)

def saveFeatureExtraction(idFeat,idNames):
    for i in range(len(idFeat)):
        np.savetxt(f'SavedFeatures//{idNames[i]}.txt',X=idFeat[i],delimiter=',')
    np.savetxt('SavedFeatures//idNames.txt',X=idNames,delimiter=',',fmt='%s')

def combineIDs(IDFeat, IDNames):
    unsortedIDs = np.copy(IDNames)
    unsortedFeats = np.copy(IDFeat)

    norm1 = np.linalg.norm(unsortedFeats[0].flatten())
    norm2 = np.linalg.norm(unsortedFeats[1].flatten())
    dist =1- distance.cosine(unsortedFeats[0].flatten() / norm1 ,unsortedFeats[1].flatten() / norm2)

    print(dist)
    #hello
    print("combine")

# img_feat, img_names = REID.extract_feature(img_dir="IDs//1.0",model_path="youtu_reid_baseline_lite.onnx",batch_size=32, resize_h=256,resize_w=128,backend=cv.dnn.DNN_BACKEND_CUDA,target=cv.dnn.DNN_TARGET_CUDA)
# print(img_feat)
# print("=========")
# print(img_names)

# idFeat, idNames = scanIDFolders("IDs")
# np.savetxt('SavedFeatures//idFeat.txt',X=idFeat,delimiter=',')
# np.savetxt('SavedFeatures//idNames.txt',X=idNames,delimiter=',')

#saveFeatureExtraction()

IDFeat, IDNames = scanIDFolders("IDs")
saveFeatureExtraction(idFeat=IDFeat,idNames=IDNames)
#visualiseExtraction(idFeat=IDFeat,idNames=IDNames)





#scanRoom()

#trainedScan()

#main()
