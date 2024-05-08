import pandas as pd
import json
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pickle
import glob
import shutil

questions = pd.DataFrame(json.load(open("./v2_OpenEnded_mscoco_train2014_questions.json", "r"))["questions"])
annotations = pd.DataFrame(json.load(open("./v2_mscoco_train2014_annotations.json", "r"))["annotations"])
compairs = json.load(open("./v2_mscoco_train2014_complementary_pairs.json"))


def IdtoFile(imageId):
    imageId = str(imageId)
    return f"COCO_train2014_000000{imageId}.jpg"

def FiletoId(filename):
    return filename.split("/")[-1][-10:-4]

def responseWeight(confidence):
    if confidence == "yes":
        return 10
    elif confidence == "maybe" :
        return 7
    return 4


def createRandomSample(p : float = 0.01):
    sample = set()
    imgfiles = os.listdir('./../RawImages/images/')[1:]
    for img in imgfiles:
        if np.random.choice([0, 1], p = [1 - p, p]):
            sample.add(FiletoId(img))
    return sample

sample = createRandomSample(0.30)
pickle.dump(sample, open("training_sample.pkl", "wb"))

def createDataset(datapath, imageset):
    files = glob.glob(datapath + "*")
    for f in files :
        os.remove(f)
    print(f"Erased contents of the {datapath} directory.")
    
    print(f"Building the data dict...")
    dict = {}
    answer_confidences = set()
    labels = []
    df = pd.merge(questions, annotations, on = "question_id")
    for i, row in tqdm(df.iterrows()):
        # if i > 100:
        #     break
        imageId = str(row["image_id_x"])
        if imageId not in imageset:
            continue
        questionText = row["question"]
        answer = row["multiple_choice_answer"]
        for response in row["answers"]:
            labels.append(response["answer"])
        
        if imageId in dict:
            dict[imageId].append({
                "question" : questionText,
                "answer" : answer,
                "responses" : row["answers"]
            })
        else : 
            dict[imageId] = [{
                "question" : questionText,
                "answer" : answer,
                "responses" : row["answers"]
            }]
    labelEncoder = LabelEncoder()
    labelEncoder.fit(labels)
    C = len(labelEncoder.classes_)
    labelmap = {}
    for i in range(C):
        labelmap[labelEncoder.classes_[i]] = i
    print(f"Label Encoder contains {C} classes.")
    
    print(f"Building the metadata object and moving files...")
    metadata = []
        
    for imageId in tqdm(dict.keys()): 
        fileName = IdtoFile(imageId)
        qna = []
        for q in dict[imageId]:
            questionText = q["question"]
            answer = q["answer"]
            answerlabel = labelmap[answer]
            # softvec = {}
            # for response in q["responses"]:
            #     label = labelmap[response["answer"]]
            #     if label in softvec:
            #         softvec[label] += responseWeight(response["answer_confidence"])
            #     else:
            #         softvec[label] = responseWeight(response["answer_confidence"])
            qna.append({
                "question" : questionText,
                "answer" : answer,
                "answerlabel" : answerlabel,
                # "softvec" : softvec
            })
        shutil.copyfile(f"../RawImages/images/{fileName}", f"./train/{fileName}")
        metadata.append({
            "file_name" : fileName,
            "qna" : qna 
        })
        with open("./train/metadata.jsonl", "w") as f:
            for item in metadata:
                json.dump(item, f)
                f.write('\n')
    
        with open("labelEncoder.pkl", "wb") as f:
            pickle.dump(labelEncoder, f)
            
    return labelEncoder, metadata

labelEncoder, metadata = createDataset(
    ".\\train\\",
    sample
)