import os
import cv2
import pandas as pd
import zipfile

with zipfile.ZipFile("test_data.zip", 'r') as zip_ref:
    zip_ref.extractall(path="test_data")

with zipfile.ZipFile("train_data.zip", 'r') as zip_ref:
    zip_ref.extractall(path="train_data")


TRAIN_CSV = "./train_data/train_data.csv"
TRAIN_IMGS_DIR = "./train_data/images/"
TEST_CSV = "./test_data/test_data.csv"
TEST_IMGS_DIR = "./test_data/images/"
OUTPUT_CSV = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)

#Get images in training dataset
for _, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMGS_DIR, row['image'])
    img = cv2.imread(img_path)

test_df = pd.read_csv(TEST_CSV)
submission_data = []

for _, row in test_df.iterrows():
    subtask_id = int(row['subtaskID'])
    datapoint_id = int(row['datapointID'])

    if subtask_id == 1: #SOLVE SUBTASK 1
        intruder = 1 #Replace with code to find intruder
        submission_data.append([subtask_id, datapoint_id, intruder]) 

    elif subtask_id == 2: #SOLVE SUBTASK 2
        angle = 0 #Replace with code to find angle
        submission_data.append([subtask_id, datapoint_id, angle])

sub_df = pd.DataFrame(submission_data, columns=['subtaskID', 'datapointID', 'answer'])
sub_df.to_csv(OUTPUT_CSV, index=False)
