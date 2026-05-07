import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

#Load data and model
test_df = pd.read_csv("test_data.csv")

model_path = os.path.expanduser("BAAI/bge-small-en-v1.5")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

#Example of using model to get sentence embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sample_text = test_df['text'].iloc[0]

with torch.no_grad():
    encoded_input = tokenizer(sample_text, return_tensors="pt").to(device)
    model_output = model(**encoded_input)

# [CLS] token at index 0 stores embedding of the whole sentence
embedding = model_output[0][:, 0]


datapoint_ids = test_df['datapointID'] 

# Subtask 1
dummy_answer_subtask1 = [0] * len(datapoint_ids) #REPLACE WITH CODE TO SOLVE SUBTASK 1
subtask1_df = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': datapoint_ids,
    'answer': dummy_answer_subtask1
})

# Subtask 2
dummy_answer_subtask2 = [1] * len(datapoint_ids) #REPLACE WITH CODE TO SOLVE SUBTASK 2

subtask2_df = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': datapoint_ids,
    'answer': dummy_answer_subtask2
})

submission_df = pd.concat([subtask1_df, subtask2_df], ignore_index=True)

submission_file = "output.csv"
submission_df.to_csv(submission_file, index=False)