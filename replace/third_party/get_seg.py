import os
import shutil
import matplotlib.pyplot as plt
import os
import re
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import torch
import numpy as np
import matplotlib
import json 
from argparse import ArgumentParser
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.models.sam import SAM3VideoSemanticPredictor
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser()
parser.add_argument(
    "--seva_folder", type=str, default=os.path.join(BASE_DIR, "..", "..", "..", "stable-virtual-camera","output"),
    help="Seva image folder (default: %(default)s)"
)
parser.add_argument(
    "--caption_path", type=str, default=os.path.join(BASE_DIR, "..", "..", "..", "captions.json"),
    help="Caption file (default: %(default)s)"
)
parser.add_argument(
    "--start_end_idx", type=str, default=None,
    help="Start and end index of the images to process (default: %(default)s)"
)

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

TYPE_ORDER = {
    "zoom": 0,
    "rotate": 1,
    "": 2,          
}

def sort_key(fname):
    # "1_0" -> [1,0]
    base = os.path.basename(fname)
    nums = [int(n) for n in re.findall(r'\d+', base)]
    
    if "zoom" in base:
        t = "zoom"
    elif "rotate" in base:
        t = "rotate"
    else:
        t = ""
    return nums + [TYPE_ORDER[t]]

def parse_pair(s: str):
    # "10_0" can be parsed to (10,0)
    nums = re.findall(r'\d+', s)
    if len(nums) < 2:
        raise ValueError(f"cannot parse i_j: {s}")
    return (int(nums[0]), int(nums[1]))


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = np.squeeze(mask)  # (1,H,W) or (1,1,H,W) to (H,W)
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def mask_to_bw(masks):
    """
    mask: torch.Tensor, shape (H, W), dtype 0/1 or bool
    return: PIL.Image
    """
    masks = masks.squeeze(1) if masks.dim() == 4 else masks
    combined = masks.sum(0) > 0  
    mask_np = combined.cpu().numpy().astype(np.uint8) * 255  
    return Image.fromarray(mask_np, mode="L")


def seg_img(args):
    # create mid_output folder
    os.makedirs("mid_output", exist_ok=True)
    save_folder = os.path.join(BASE_DIR, "..", "..", "mid_output", "foreground_images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # else:
    #     for name in os.listdir(save_folder):
    #         path = os.path.join(save_folder, name)
    #         if os.path.isfile(path):
    #             os.remove(path)
    #         elif os.path.isdir(path):
    #             shutil.rmtree(path)
    input_fg_folder = os.path.join(save_folder, "input_images")
    os.makedirs(input_fg_folder, exist_ok=True)
    seva_fg_folder = os.path.join(save_folder, "seva_images")
    os.makedirs(seva_fg_folder, exist_ok=True)
    
    # load input images
    with open(args.caption_path, "rb") as f:
        img_dict = json.load(f)
    img_dict_list = list(img_dict.items())
    all_file = []
    all_fg = []
    for path_, info in img_dict_list:
        all_file.append(path_)
        all_fg.append(info["foreground"])
    input_images = []
    input_fg_list = []
    if args.start_end_idx is not None:
        # 0_0.png,10_1.png
        start_name, end_name = [s.strip() for s in args.start_end_idx.split(",")]
        names = [os.path.basename(f) for f in all_file]
        try:
            start_i = names.index(start_name)
            end_i = names.index(end_name)
        except ValueError as e:
            raise ValueError(f"start or end file not found: {e}")
        if start_i > end_i:
            raise ValueError("start_end_idx order is invalid (start > end)")

        input_images = all_file[start_i : end_i + 1]
        input_fg_list = all_fg[start_i : end_i + 1]
        names = names[start_i : end_i + 1]
    else:
        input_images = all_file
        input_fg_list = all_fg
        names = [os.path.basename(f) for f in all_file]
    
    seva_fg_list = []
    # if args.traj_json is not None:
    #     for path_ in data_paths:
    #         traj_json_path = os.path.join(path_, args.traj_json)
    #         with open(traj_json_path, "r") as f:
    #             traj_json_data = json.load(f)
    #         temp_list.extend(list(traj_json_data.items()))
    #     # check if names is in traj_json_list key
    #     # sort traj_json_list by name
    #     temp_list.sort(key=lambda x: sort_key(x[0]))
    #     assert len(temp_list) == len(all_file), f"traj_json element number {len(temp_list)} and img_file length {len(all_file)} mismatch"
    #     # select element only in start_end_idx range
    #     if args.start_end_idx is not None:
    #         temp_list = temp_list[start_i : end_i + 1]
    #         fg_dict_list = fg_dict_list[start_i : end_i + 1]
    #     traj_json_names = [name for name, _ in temp_list]
        
    #     for name in names:
    #         if name not in traj_json_names:
    #             raise ValueError(f"name {name} not found in traj_json")
        
    #     for i in range(len(temp_list)):
    #         name, info = temp_list[i]
    #         _, fg_info = fg_dict_list[i]
    #         input_fg_list.append(fg_info["foreground"])
    #         if len(info.get("traj_prior").split(",")) > 1:
    #             seva_fg_list.append(fg_info["foreground"])
    #             seva_fg_list.append(fg_info["foreground"])
    #         else:
    #             seva_fg_list.append(fg_info["foreground"])
                
    # else:
    #     for _, fg_info in fg_dict_list:
    #         input_fg_list.append(fg_info["foreground"])
    #         seva_fg_list.append(fg_info["foreground"])
        
    
    seva_folder = args.seva_folder
    if args.start_end_idx is not None:
        seva_img_folders = [
            f for f in os.listdir(seva_folder) if not f.startswith('.') and os.path.isdir(os.path.join(seva_folder, f))
        ]
        seva_img_folders = sorted(seva_img_folders, key=sort_key)
        start, end = [s.strip().split(".")[0] for s in args.start_end_idx.split(",")]
        start_pair = parse_pair(start)   # (0,0)
        end_pair   = parse_pair(end)     # (10,0)
        seva_img_folders = [
            f for f in seva_img_folders
            if start_pair <= parse_pair(f) <= end_pair
        ]
    else:
        seva_img_folders = [
            f for f in os.listdir(seva_folder) if not f.startswith('.') and os.path.isdir(os.path.join(seva_folder, f))
        ]
        seva_img_folders = sorted(seva_img_folders, key=sort_key)
    

    folder_tuple = [parse_pair(f) for f in seva_img_folders]
    i = 0
    j = 0
    while i < len(seva_img_folders):
        if i+1 < len(seva_img_folders) and folder_tuple[i+1] == folder_tuple[i]:
            seva_fg_list.append(input_fg_list[j])
            seva_fg_list.append(input_fg_list[j])
            j += 1
            i += 2     
        else:
            seva_fg_list.append(input_fg_list[j])
            j += 1
            i += 1
    
    # sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    # bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    # sam_model = build_sam3_image_model(bpe_path=bpe_path)
    # sam3_processor = Sam3Processor(sam_model, confidence_threshold=0.5)
    
    overrides = dict(
    conf=0.35,
    task="segment",
    mode="predict",
    imgsz=512,
    model=f"{BASE_DIR}/sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=False,
    exist_ok=True,
    verbose=False
    )
    predictorSemantic = SAM3SemanticPredictor(overrides=overrides)
    predictorVideo = SAM3VideoSemanticPredictor(overrides=overrides)
    

    for img_folder in seva_img_folders:
        os.makedirs(os.path.join(seva_fg_folder, img_folder), exist_ok=True)

    # seg input img
    for i, (image_path, fg, name) in enumerate(zip(input_images, input_fg_list, names)):
        print(f"processing input image {i+1}/{len(input_images)}")
        # Load an image
        image = Image.open(image_path).convert("RGB")
        predictorSemantic.set_image(image)
        result = predictorSemantic(text=[fg]) 
                #  bboxes=[[38, 38, 474, 474]]
        # masked_img = overlay_masks(image, masks)
        # masked_img.save("masked_img.png")
        masks = result[0].masks
        # mask have data attribute
        if hasattr(masks, "data"):
            masks = masks.data
        else:
            masks = torch.zeros(image.size[1], image.size[0])
            masks = masks.unsqueeze(0).unsqueeze(0)
        masked_img = mask_to_bw(masks)
        masked_img.save(f"{input_fg_folder}/{name}")
    
    # seg seva img
    img_temp = Image.open(input_images[0]).convert("RGB")
    
    for i in range(len(seva_img_folders)):
        print(f"processing seva image {i+1}/{len(seva_img_folders)}")
        print(seva_fg_list[i])
        img_folder = os.path.join(seva_folder, seva_img_folders[i])
        video_path = os.path.join(img_folder, "samples-rgb.mp4")
        
        results = predictorVideo(source=video_path, text=[seva_fg_list[i]], stream=True)
        for j, r in enumerate(results):
            masks = r.masks
            if hasattr(masks, "data"):
                masks = masks.data
            else:
                masks = torch.zeros(img_temp.size[1], img_temp.size[0])
                masks = masks.unsqueeze(0).unsqueeze(0)
            masked_img = mask_to_bw(masks)
            masked_img.save(f"{seva_fg_folder}/{seva_img_folders[i]}/{j:03d}.png")
        predictorVideo.inference_state.clear()
        
   
    
if __name__ == "__main__":
    args = parser.parse_args()
    seg_img(args)