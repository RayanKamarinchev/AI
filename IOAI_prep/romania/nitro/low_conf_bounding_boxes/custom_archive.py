#######################################################################################################################
############################################# UNZIPPING the ARCHIVES ##################################################
#######################################################################################################################
import zipfile


with zipfile.ZipFile("test_data.zip", 'r') as zip_ref:
    zip_ref.extractall(path="")

with zipfile.ZipFile("train_data.zip", 'r') as zip_ref:
    zip_ref.extractall(path="")

# !unzip train_data.zip # in jupyter notebook, you can use this command to unzip directly in the cell
# !unzip test_data.zip # in jupyter notebook, you can use this command to unzip directly in the cell.

#######################################################################################################################
####################### Reading, processing, and writing a CSV files for a valid submission. ##########################
#######################################################################################################################

import os
import csv
import json
import torch

ROOT = "./drive/MyDrive/NATIONALA_2026" # Change this to your dataset path
TRAIN_CSV = os.path.join(ROOT, "train", "train.csv")
TEST_PRED_CSV = os.path.join(ROOT, "test", "test.csv")
SUBMISSION_CSV = "submission_dummy.csv"

def my_nms_implementation_but_should_not_be_yours(boxes, scores, iou_thresh=0.05):
    # This is a placeholder for your NMS implementation.
    # Replace this with your actual NMS logic.
    # In this snippet, the function simply returns all indices, which means no boxes are filtered out.
    # You should implement your NMS logic here to return only the indices of boxes that survive the suppression.
    return torch.arange(len(boxes))

def create_submission(test_csv_path, output_csv_path, nms_iou_threshold=0.05):
    submission_rows = []
    
    # 1. Read the provided test csv with Faster R-CNN predictions
    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    # 2. Iterate through rows. The index in this loop becomes the datapointID!
    for idx, row in enumerate(reader):
        # Parse strings into tensors

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # It's important to call json.loads(...) to convert the string representation of lists 
        # back into actual lists before converting to tensors.

        boxes = torch.tensor(json.loads(row["pred_boxes"]), dtype=torch.float32) 
        scores = torch.tensor(json.loads(row["pred_scores"]), dtype=torch.float32)

        # =========================================================================================
        # UNDERSTANDING THE DETECTOR'S OUTPUT:
        # 
        # 'pred_boxes': A list of bounding boxes predicted by the Faster R-CNN model. 
        #               Each box is represented as [x_min, y_min, x_max, y_max], which are the 
        #               pixel coordinates of the top-left and bottom-right corners.
        #
        # 'pred_scores': The AI's confidence score (ranging from 0.0 to 1.0) for each box. 
        #                A score of 0.95 means the AI is 95% sure there is a person there.
        #
        # REMEMBER: Because Professor Grammar set the threshold extremely low (0.05), the model 
        #           is "over-predicting." You will see many overlapping boxes for the same person, 
        #           and many low-scoring boxes on random background objects. 
        #           Your goal is to use NMS (and later, whatever you want) to filter this list down!
        # You might want to overlay these boxes on the original images to visually understand what the model is predicting and 
        #           how NMS is helping to clean it up. SEE EXTRA DETAILS AT THE END OF THIS SCRIPT. 
        # USING THE IMAGES IS NOT REQUIRED FOR YOUR ALGORITHM!       
        # =========================================================================================
        
        # Ensure they are sorted by score descending before running NMS
        if scores.numel() > 0:
            sort_idx = scores.argsort(descending=True)
            boxes = boxes[sort_idx]
            scores = scores[sort_idx]
        
        # Apply NMS
        keep_indices = my_nms_implementation_but_should_not_be_yours(boxes, scores, iou_thresh=nms_iou_threshold)
        
        # Get final boxes and format them back to a JSON string
        final_boxes = boxes[keep_indices].tolist()
        answer_str = json.dumps(final_boxes) # !!!!!!!!!!!!!! very important to convert back to string for CSV output!
        
        # Append for Subtask 1
        submission_rows.append([1, idx, answer_str])
        
        # Append for Subtask 2 (using the same NMS logic for both to get a valid file)
        # In a real scenario, you would run your different algorithm for subtask 2 here and 
        # append those results instead.
        submission_rows.append([2, idx, answer_str])

    # 3. Write out the final CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subtaskID", "datapointID", "answer"])
        writer.writerows(submission_rows)
        
    print(f"Successfully created {output_csv_path} with {len(submission_rows)} rows!")

#######################################################################################################################
####################### Functions to use if you want to perform local evaluation. ##########################
#######################################################################################################################

import numpy as np

import numpy as np
import json
import csv

def box_iou_np(boxes1, boxes2):
    """
    Calculates Intersection over Union (IoU) between two sets of boxes.
    boxes1: numpy array of shape (N, 4) in format [x1, y1, x2, y2]
    boxes2: numpy array of shape (M, 4) in format [x1, y1, x2, y2]
    Returns: numpy array of shape (N, M) containing IoU scores.
    """
    boxes1 = np.array(boxes1, dtype=np.float32)
    boxes2 = np.array(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0, boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # top-left
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # bottom-right

    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def evaluate_detection_np(pred_boxes, gt_boxes, iou_thr=0.5):
    """
    Evaluates one image using greedy one-to-one matching.

    IMPORTANT:
    pred_boxes must be sorted by confidence score (highest to lowest).

    Returns a dictionary with:
      tp, fp, fn, precision, recall, f1, f2
    """
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)

    # Handle edge cases
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0,
            "f1": 1.0, "f2": 1.0
        }

    if len(pred_boxes) == 0:
        fn = len(gt_boxes)
        return {
            "tp": 0, "fp": 0, "fn": fn,
            "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "f2": 0.0
        }

    if len(gt_boxes) == 0:
        fp = len(pred_boxes)
        return {
            "tp": 0, "fp": fp, "fn": 0,
            "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "f2": 0.0
        }

    ious = box_iou_np(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)

    tp = 0
    fp = 0

    # Greedy matching
    for p in range(len(pred_boxes)):
        row = ious[p].copy()
        row[gt_used] = -1.0

        best_g = np.argmax(row)
        best_iou = row[best_g]

        if best_iou >= iou_thr:
            tp += 1
            gt_used[best_g] = True
        else:
            fp += 1

    fn = np.sum(~gt_used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 and recall == 0:
        f1 = 0.0
        f2 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        f2 = (1 + 2**2) * precision * recall / (2**2 * precision + recall)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
    }


def evaluate_train_baseline(train_csv_path):
    print(f"Loading training data from: {train_csv_path}")
    with open(train_csv_path, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    f1_scores = []
    f2_scores = []

    print("Evaluating baseline (Score-Only) performance on training set...")

    for row in train_rows:
        pred_boxes_list = json.loads(row["pred_boxes"])
        pred_scores_list = json.loads(row["pred_scores"])
        gt_boxes_list = json.loads(row["gt_boxes"])

        # Sort predicted boxes by score descending
        if len(pred_scores_list) > 0:
            paired = list(zip(pred_scores_list, pred_boxes_list))
            paired.sort(key=lambda x: x[0], reverse=True)
            pred_boxes_list = [box for score, box in paired]

        result = evaluate_detection_np(pred_boxes_list, gt_boxes_list, iou_thr=0.5)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        f1_scores.append(result["f1"])
        f2_scores.append(result["f2"])

    # Global precision / recall / F1 / F2
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if global_precision == 0 and global_recall == 0:
        global_f1 = 0.0
        global_f2 = 0.0
    else:
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall)
        global_f2 = (1 + 2**2) * global_precision * global_recall / (2**2 * global_precision + global_recall)

    # !!!! Here is the official score that will be used for ranking in the competition
    official_score = 0.4 * global_f2 + 0.6 * global_f1

    print("-" * 50)
    print("TRAIN SET BASELINE RESULTS (Score threshold = 0.05)")
    print("-" * 50)
    print(f"Total True Positives (TP):   {total_tp}")
    print(f"Total False Positives (FP):  {total_fp}")
    print(f"Total False Negatives (FN):  {total_fn}")
    print(f"Global Precision:            {global_precision:.4f}")
    print(f"Global Recall:               {global_recall:.4f}")
    print(f"Global F1@0.5:               {global_f1:.4f}")
    print(f"Global F2@0.5:               {global_f2:.4f}")
    print(f"Mean image-level F1@0.5:     {np.mean(f1_scores):.4f}")
    print(f"Mean image-level F2@0.5:     {np.mean(f2_scores):.4f}")
    print("-" * 50)
    print("If your NMS / filtering algorithm is working, FP should usually go down.")

    print("-" * 50)
    print(f"Official score (for leaderboard):  {official_score:.4f}")
    print("-" * 50)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_boxes(image_path, boxes, scores=None, color='red', title="Bounding Boxes"):
    """
    Reads an image and overlays a list of bounding boxes.
    
    Parameters:
    - image_path (str): Path to the image file.
    - boxes (list or array): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    - scores (list or array, optional): Confidence scores to display above each box.
    - color (str, optional): Color of the bounding boxes (e.g., 'red', 'green', 'blue').
    - title (str, optional): Title for the plot.
    """
    try:
        # Open the image
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(title)
    
    # Loop over the boxes and add them to the plot
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # If scores are provided, add them as text slightly above the box
        if scores is not None and i < len(scores):
            score_text = f"{scores[i]:.2f}"
            ax.text(x_min, y_min - 5, score_text, color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none'))
            
    # Remove axes ticks for a cleaner look
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# You can call this at the bottom of the script:
if __name__ == "__main__":
    # TEST_CSV_PATH = os.path.join(ROOT, "test", "test_predictions.csv")
    # SUBMISSION_PATH = "submission_dummy.csv"
    # create_submission(TEST_CSV_PATH, SUBMISSION_PATH, nms_iou_threshold=0.05)
    
    # Test the evaluator
    evaluate_train_baseline(TRAIN_CSV)

    # Example usage of the visualization function

    # =========================================================================================
    # EXAMPLE: Visualizing the raw detector output vs the Ground Truth for a specific image
    # =========================================================================================
    # image_file = "FudanPed00015.png" # Pick a filename you see inside your train.csv!
    # img_path = os.path.join(ROOT, "train", "images", image_file)
    #
    # # Load the corresponding row from the train csv
    # with open(TRAIN_CSV, "r", encoding="utf-8") as f:
    #     for row in csv.DictReader(f):
    #         if row["filename"] == image_file:
    #             pred_boxes_list = json.loads(row["pred_boxes"])
    #             pred_scores_list = json.loads(row["pred_scores"])
    #             gt_boxes_list = json.loads(row["gt_boxes"])
    #             break
    #
    # # Visualize the Raw Predictions (Too many boxes!)
    # visualize_boxes(img_path, pred_boxes_list, scores=pred_scores_list, color='red', title="Raw Detector Output (Professor Grammar's Mess)")
    #
    # # Visualize the Ground Truth (What it should look like!)
    # visualize_boxes(img_path, gt_boxes_list, color='green', title="Ground Truth (The Actual Students)")
    # =========================================================================================