import os
import torch
import numpy as np
import cv2
import re
from PIL import Image
from argparse import ArgumentParser
from model import DiffMorpherPipeline
import json
import shutil
import torchvision.transforms as T
from torchvision.utils import save_image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser()
parser.add_argument("--caption_path", type=str, default=os.path.join(BASE_DIR, "..", "captions.json"),
                    help="Caption file (default: %(default)s)")
parser.add_argument("--seva_folder", type=str, default=os.path.join(BASE_DIR, "..", "stable-virtual-camera","output"),
                    help="Seva image folder (default: %(default)s)")
parser.add_argument("--start_end_idx", type=str, default=None,
                    help="Start and end index of the images to process (default: %(default)s)")

parser.add_argument(
    "--model_path", type=str, default="Manojb/stable-diffusion-2-1-base",
    help="Pretrained model to use (default: %(default)s)"
)

parser.add_argument(
    "--save_lora_dir", type=str, default="./lora",
    help="Path of the output lora directory (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_0", type=str, default="",
    help="Path of the lora directory of the first image (default: %(default)s)"
)
parser.add_argument(
    "--load_lora_path_1", type=str, default="",
    help="Path of the lora directory of the second image (default: %(default)s)"
)
parser.add_argument(
    "--use_adain", default=True,
    help="Use AdaIN (default: %(default)s)"
)
parser.add_argument(
    "--use_reschedule", default=True,
    help="Use reschedule sampling (default: %(default)s)"
)
parser.add_argument(
    "--lamb",  type=float, default=0.0, # 0.6, 0.0
    help="Lambda for self-attention replacement (default: %(default)s)"
)
parser.add_argument(
    "--fix_lora_value", type=float, default=None,
    help="Fix lora value (default: LoRA Interp., not fixed)"
)
parser.add_argument(
    "--save_inter", default=False,
    help="Save intermediate results (default: %(default)s)"
)
parser.add_argument(
    "--num_frames", type=int, default=30,
    help="Number of frames to generate (default: %(default)s)"
)
parser.add_argument(
    "--duration", type=int, default=100,
    help="Duration of each frame (default: %(default)s ms)"
)
parser.add_argument(
    "--no_lora", action="store_true"
)


args = parser.parse_args()

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


with open(os.path.join(args.caption_path), "rb") as f:
    data = json.load(f)
items = list(data.items())
image_path = []
prompts = []
for path_, info in items:
    image_path.append(path_)
    prompts.append(info["caption"])

if args.start_end_idx is not None:
    # 0_0.png,10_1.png
    start_name, end_name = [s.strip() for s in args.start_end_idx.split(",")]
    names = [os.path.basename(f) for f in image_path]
    start_i = names.index(start_name)
    end_i = names.index(end_name)
    image_path = image_path[start_i : end_i + 1]
    prompts = prompts[start_i : end_i + 1]
    
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# else:
#     for name in os.listdir(output_folder):
#         path = os.path.join(output_folder, name)
#         if os.path.isfile(path):
#             os.remove(path)
#         elif os.path.isdir(path):
#             shutil.rmtree(path)

pipeline = DiffMorpherPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.float32)
pipeline.to("cuda")


entry_folder = args.seva_folder
sample_folder = "samples-rgb"

img_folders = [
    f for f in os.listdir(entry_folder) if not f.startswith('.') and os.path.isdir(os.path.join(entry_folder, f))
]
img_folders = sorted(img_folders, key=sort_key)

if args.start_end_idx is not None:
    start, end = [s.strip().split(".")[0] for s in args.start_end_idx.split(",")]
    start_pair = parse_pair(start)   # (0,0)
    end_pair   = parse_pair(end)     # (10,0)
    img_folders = [
        f for f in img_folders
        if start_pair <= parse_pair(f) <= end_pair
    ]
folder_tuple = [parse_pair(f) for f in img_folders]

search_file = os.path.join(BASE_DIR, "mid_output", "sparse_matching_json", "search.json")
with open(search_file, "r") as f:
    search_dict = json.load(f)
search_dict_list = list(search_dict.items())
search_dict_list = sorted(search_dict_list, key=lambda x: sort_key(x[0]))

search_list = []
name_list = []
for name, search in search_dict_list:
    name_list.append(name)
    search_list.append(search)
if args.start_end_idx is not None:
    start, end = [s.strip().split(".")[0] for s in args.start_end_idx.split(",")]
    start_pair = parse_pair(start)   # (0,0)
    end_pair   = parse_pair(end)     # (10,0)
    temp_list = [
        f for f in name_list
        if start_pair <= parse_pair(f) <= end_pair
    ]
    start_i = name_list.index(temp_list[0])
    end_i = name_list.index(temp_list[-1])
    search_list = search_list[start_i : end_i + 1]
    
assert len(search_list) == len(img_folders), f"search_list length {len(search_list)} and img_folders length {len(img_folders)} mismatch"

flag = 0
temp_list = []
temp_search = []
search0 = []
search1 = []
augmented_img_list = []
i = 0
while i < len(img_folders):
    if i + 1 < len(img_folders) and folder_tuple[i+1] == folder_tuple[i]:
        j = 0
        while j < 4:
            img_folder = os.path.join(entry_folder, img_folders[i], sample_folder)
            img_list = [
                f for f in os.listdir(img_folder) if not f.startswith('.') and os.path.isfile(os.path.join(img_folder, f))
            ]
            img_list = sorted(img_list, key=sort_key)
            img_list = [os.path.join(img_folder, img) for img in img_list]
            temp_list.extend(img_list)
            temp_search.extend(search_list[i])
            if j == 1:
                search0.append(temp_search)
                temp_search = []
            if j == 3:
                search1.append(temp_search)
                temp_search = []
            j += 1
            i += 1
        augmented_img_list.append(temp_list)
        temp_list = []
    else:
        img_folder = os.path.join(entry_folder, img_folders[i], sample_folder)
        img_list = [
            f for f in os.listdir(img_folder) if not f.startswith('.') and os.path.isfile(os.path.join(img_folder, f))
        ]
        img_list = sorted(img_list, key=sort_key)
        img_list = [os.path.join(img_folder, img) for img in img_list]
        temp_list.extend(img_list)
        if flag == 0:
            search0.append(search_list[i])
        else:
            search1.append(search_list[i])
        flag += 1
        if flag == 2:
            augmented_img_list.append(temp_list)
            temp_list = []
            flag = 0
        i += 1
        
print(f"no lora: {args.no_lora}")
for idx, (path0, path1, cap0, cap1) in enumerate(zip(image_path[::2], image_path[1::2], prompts[::2], prompts[1::2])):
    print(path0, cap0, search0[idx])
    print(path1, cap1, search1[idx])
    images, input_seva_latents, mid_latents = pipeline(
        img_path_0=path0,
        img_path_1=path1,
        augmented_img_list=augmented_img_list[idx],
        train_batch_size=20,
        prompt_0=cap0,
        prompt_1=cap1,
        search_0=search0[idx],
        search_1=search1[idx],
        save_lora_dir=args.save_lora_dir,
        load_lora_path_0=args.load_lora_path_0,
        load_lora_path_1=args.load_lora_path_1,
        lora_steps=200, # 200 80 40 
        lora_lr=2e-4, # 2e-4
        lora_rank=16,
        use_adain=args.use_adain,
        use_reschedule=args.use_reschedule,
        lamd=args.lamb,
        output_path=output_folder,
        num_frames=args.num_frames,
        fix_lora=args.fix_lora_value,
        save_intermediates=args.save_inter,
        use_lora=not args.no_lora,
        index = idx,
        if_test_pca=True,
        )
    
    pil_to_tensor = T.ToTensor()
    tensor_images = [pil_to_tensor(img) for img in images]
    res = torch.stack(tensor_images, dim=0)
    save_name = os.path.basename(path0).split('.')[0].split('_')[0]
    save_image(res, f"{output_folder}/diffmorph_{save_name}.png")
    
    # save distribution of x2_input_seva and x2_mid_seq
    
    np.save(f"{output_folder}/lora_1_{save_name}.pth", input_seva_latents)
    np.save(f"{output_folder}/lora_2_{save_name}.pth", mid_latents)

    
    # images[0].save(f"{args.output_path}/output.gif", save_all=True,
    #             append_images=images[1:], duration=args.duration, loop=0)
