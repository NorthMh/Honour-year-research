from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2Processor
import torch
from PIL import Image
import os
import json
import re
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default=os.path.join(BASE_DIR, "..", "images"),
                    help="Image folder (default: %(default)s)")
parser.add_argument("--caption_path", type=str, default=os.path.join(BASE_DIR, "..", "captions.json"),
                    help="Caption file (default: %(default)s)")
args = parser.parse_args()

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

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_folder = []
args.image_folder = os.path.abspath(args.image_folder)
if os.path.isdir(args.image_folder):
    subfolders = [f for f in os.listdir(args.image_folder) if os.path.isdir(os.path.join(args.image_folder, f)) and not f.startswith('.')]
    if len(subfolders) == 0:
        print(f"No subfolders found in {args.image_folder}")
        image_folder = [args.image_folder]
    else:
        print(f"Subfolders found in {args.image_folder}: {subfolders}")
        image_folder = [os.path.join(args.image_folder, subfolder) for subfolder in subfolders]
else:
    raise ValueError(f"Data path is not a folder: {args.image_folder}")
image_index_list = []
for folder in image_folder:
    for f in os.listdir(folder):
        if not f.startswith('.') and os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png')):
            image_index_list.append(os.path.join(folder, f))
image_index_list = sorted(image_index_list, key=sort_key)

image_list = [Image.open(idx).convert("RGB") for idx in image_index_list]
image_dict = {}

inputs = processor(image_list, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_new_tokens=40)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
length = len(generated_text)


qtext = f"Question: Which foreground object is in this image? Answer with a short noun phrase only. No adjectives. Answer:"
questions = [qtext] * len(image_list)

inputs_ = processor(image_list, questions, return_tensors="pt").to(device)
out_ids = model.generate(**inputs_, max_new_tokens=5)
foreground_list = processor.batch_decode(out_ids, skip_special_tokens=True)
# foreground_list = processor.decode(out_ids[0], skip_special_tokens=True)
foreground_list = [ f.split("Answer:", 1)[-1].strip() for f in foreground_list]
print(foreground_list)
for i in range(length):
    # print(f"[{i}] {t.strip()}")
    image_dict[image_index_list[i]] = {
        "caption": generated_text[i].strip(),
        "foreground": foreground_list[i].strip()
    }
print(f"image_dict: {image_dict}")

with open(args.caption_path, "w") as f:
    json.dump(image_dict, f)

# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# from PIL import Image
# import os
# import json
# import re

# def sort_key(fname):
#     # "1_0" -> [1,0]
#     base = os.path.basename(fname)
#     nums = re.findall(r'(\d+)', base)
#     return [int(n) for n in nums]

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
# model.to(device)

# image_folder = f"{BASE_DIR}/../images"
# image_index_list = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png'))]
# image_index_list = sorted(image_index_list, key=sort_key)

# image_list = [Image.open(idx) for idx in image_index_list]
# image_dict = {}
# image_dict = {}
# text = "a photography of"

# for image_path in image_index_list:
#     img = Image.open(image_path)
#     inputs = processor(img, text, return_tensors="pt").to(device)
#     out = model.generate(**inputs)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     image_dict[image_path] = caption.strip()

# print(f"image_dict: {image_dict}")


# with open(f"{BASE_DIR}/../captions.json", "w") as f:
#     json.dump(image_dict, f)