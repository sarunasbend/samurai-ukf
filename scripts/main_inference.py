import cv2
import gc
import numpy as np
import os
import os.path as osp
import pdb
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

IOU_THRESHOLD = 0.1
SAVE_TO_VIDEO = True

def load_lasot_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    
    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        x, y, w, h = map(int, line.split(','))
        prompts[fid] = ((x, y, x+w, y+h), 0)
        fid += 1

    return prompts

# same IoU calculation from Kalman Filter module
def compute_iou(bbox1, bbox2):
    bbox1 = xywh_to_xyxy(bbox1) 
    bbox2 = xywh_to_xyxy(bbox2)

    if bbox2 == [0, 0, 0, 0]:
        return 0
    
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    
    # Calculate intersection area
    intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
    
    # Calculate union area
    union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x,y,x+w,y+h]

color = [
    (255, 0, 0)
]

# TODO : Add Dirs
test_txt = ""
video_root = ""
pred_folder = ""
vis_folder = ""
exp_name = "samurai"
model_name = "base_plus"

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

save_to_video = True
os.makedirs(pred_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

unfiltered_test_videos = sorted(os.listdir(video_root))
test_videos = []

with open(osp.join(video_root, test_txt), 'r') as f:
    for line in f:
        line = line.rstrip()
        if line in unfiltered_test_videos:
            test_videos.append(line)

for vid, video in enumerate(test_videos):
    frame_folder = osp.join(video_root, video, "img1")
    gt_path = osp.join(video_root, video, "gt/gt.txt")

    if not osp.exists(gt_path):
        print(f'{video} not found/invalid')
        continue

    num_frames = len(os.listdir(frame_folder))
    
    print(f"\033[91mRunning video [{vid+1}/{len(test_videos)}]: {video} with {num_frames} frames\033[0m")
    
    height, width = cv2.imread(osp.join(frame_folder, "000001.jpg")).shape[:2]

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    predictions = []

    if SAVE_TO_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            osp.join(vis_folder, f'{video}.mp4'),
            fourcc,
            30,
            (width, height)
        )

    # Start processing frames
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        prompts, truths = load_lasot_gt(gt_path)

        bbox, track_label = prompts[list(prompts.keys())[0]]
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        iou_history = {}
        failure_count = 0

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask


            bbox1 = bbox_to_vis[object_ids[0]]

            if (frame_idx + 1) in truths:
                bbox2 = truths[frame_idx + 1]
                iou = compute_iou(bbox1, bbox2)
                iou_history[frame_idx] = iou

                if iou < IOU_THRESHOLD:
                    failure_count += 1
                    print(f'Low IoU Detected {iou} in frame [{frame_idx}]...')

            predictions.append(bbox_to_vis)   

            if SAVE_TO_VIDEO:
                img = cv2.imread(
                    osp.join(frame_folder, f"{frame_idx+1:06d}.jpg")
                )

                for obj_id in mask_to_vis:
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask_to_vis[obj_id]] = color[0]
                    img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)

                for obj_id in bbox_to_vis:
                    x, y, w, h = bbox_to_vis[obj_id]
                    cv2.rectangle(img, (x, y), (x+w, y+h), color[0], 2)

                out.write(img)

    # records bbox predictions
    with open(osp.join(pred_folder, f'{video}.txt'), 'w') as f:
        for fid, pred in enumerate(predictions):
            x, y, w, h = pred[0]
            f.write(f"{fid+1},{x},{y},{w},{h}\n")

    # records the number of fails to txt file
    with open(osp.join(pred_folder, f'{video}_IoU.txt'), 'w') as f:
        for fid in iou_history:
            f.write(f"{fid},{iou_history[fid]}\n")
        f.write(f"Failures : {failure_count}\n")

    if SAVE_TO_VIDEO:
        out.release() 

    del predictor
    del state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()