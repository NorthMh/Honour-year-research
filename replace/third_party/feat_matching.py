# reference: https://github.com/facebookresearch/dinov3/blob/main/notebooks/dense_sparse_matching.ipynb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torch.nn.functional as F
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed
from torchvision import transforms
import numpy as np
from matplotlib.patches import ConnectionPatch
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re
import shutil
from argparse import ArgumentParser
parser = ArgumentParser()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
parser.add_argument("--caption_path", type=str, default=os.path.join(BASE_DIR, "..", "..", "..", "captions.json"),
                    help="Caption file (default: %(default)s)")
parser.add_argument("--seva_folder", type=str, default=os.path.join(BASE_DIR, "..", "..", "..", "stable-virtual-camera","output"),
                    help="Seva image folder (default: %(default)s)")
parser.add_argument("--start_end_idx", type=str, default=None,
                    help="Start and end index of the images to process (default: %(default)s)")
# parser.add_argument("--traj_json", type=str, default="traj.json",
#                     help="Trajectory json file (default: %(default)s)")

def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc

def compute_distances_l2(X, Y, X_squared_norm, Y_squared_norm):
    distances = -2 * X @ Y.T
    distances.add_(X_squared_norm[:, None]).add_(Y_squared_norm[None, :])
    return distances

def stratify_points(pts_2d: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    """
    stratify the points by the distance, to keep the sparse and diverse points
    args:
        pts_2d: pixel-center coordinates of the foreground patches in the left image
        threshold: the threshold for the distance
    return:
        indices_to_exclude: the indices of the foreground patches to be excluded
        indices_to_keep: the indices of the foreground patches to be kept
    """
    n = len(pts_2d) 
    max_value = threshold + 1
    pts_2d_sq_norms = torch.linalg.vector_norm(pts_2d, dim=1) # X^2 + Y^2
    pts_2d_sq_norms.square_()
    distances = compute_distances_l2(pts_2d, pts_2d, pts_2d_sq_norms, pts_2d_sq_norms) # ||X - Y||^2, (N, N)
    distances.fill_diagonal_(max_value) # distances[i][i] = max_value -> distances_mask[i][i] = False
    distances_mask = torch.empty((n, n), dtype=pts_2d.dtype, device=pts_2d.device) # boolean matrix, True if distances <= threshold
    torch.le(distances, threshold, out=distances_mask)
    ones_vec = torch.ones(n, device=pts_2d.device, dtype=pts_2d.dtype)
    # torch.mv: matrix-vector multiplication, calculate the sum of each row
    counts_vec = torch.mv(distances_mask, ones_vec) # counts_vec[i] = the number of close neighbors of the i-th point
    indices_mask = np.ones(n)
    while torch.any(counts_vec).item():
        index_max = torch.argmax(counts_vec).item()
        # delete the "most crowded" point in the crowds
        indices_mask[index_max] = 0
        # set the distance to the max value, indicating deletion
        distances[index_max, :] = max_value
        distances[:, index_max] = max_value
        # recalculate the distance mask and the  neighbors
        torch.le(distances, threshold, out=distances_mask)
        torch.mv(distances_mask, ones_vec, out=counts_vec)
    indices_to_exclude = np.nonzero(indices_mask == 0)[0]
    indices_to_keep = np.nonzero(indices_mask > 0)[0]
    return indices_to_exclude, indices_to_keep

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


STRATIFY_DISTANCE_THRESHOLD = 32.0 # threshold for the distance between two patches
MASK_FG_THRESHOLD = 0.5 # whether the patch is a foreground patch

def compute_geo_error(j, pair, seva_img_list, seva_mask_list, input_img, input_mask_tensor, select_points, patch_size, scaler, 
                      img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder = None, input_img_name = None, index = None):
    seva_img_name = seva_img_list[j].split('/')[-1].split('.')[0]
    seva_img_path = seva_img_list[j]
    seva_mask_path = seva_mask_list[j]
    seva_img = resize(Image.open(seva_img_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)
    seva_mask = Image.open(seva_mask_path).convert('L')
    seva_mask_tensor = transform(seva_mask)
    seva_mask_tensor = patch_quant_filter(seva_mask_tensor)
    
    
    # feat dim: [1, 768, 60, 60]
    with torch.no_grad():
        if pair == 0:
            feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=seva_img)
            feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=input_img)
        else:
            feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=input_img)
            feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=seva_img)
            
        feat1 = feat1.squeeze(0)
        feat2 = feat2.squeeze(0)

        # dot product of unit vector equals to cosine similarity
        feat1 = F.normalize(feat1, p=2, dim=0)
        feat2 = F.normalize(feat2, p=2, dim=0)
        # similarity_cal dim: [patch_num_1 , patch_H_2, patch_W_2]
        similarity_cal = torch.einsum("k f, f h w -> k h w", feat1.reshape(feat_dim, -1).permute(1, 0), feat2)
    
        # compute 2D patch locations in image 1
        patch_num_1 = feat1.shape[1] * feat1.shape[2]
        index_num_1 = torch.arange(patch_num_1)
        position_1 = (
            torch.stack(
                (
                    index_num_1 // feat1.shape[2],  # row index
                    index_num_1 % feat1.shape[2]    # col index
                ),
                dim = -1
            ) + 0.5  # +0.5 to get the center of the patch
        ) * patch_size #  multiply by patch_size to obtain the center coordinates of each patch in the original image
        position_1 = position_1.to('cuda')
        
        # For each Patch in Image 1, find the Patch with the most similar features in Image 2
        similarity_flat = torch.flatten(similarity_cal, start_dim=-2)  # [patch_num_1, patch_num_2]
        index_num_2 = similarity_flat.argmax(dim=-1)  # find the index of the patch with the most similar features in Image 2, return the index, [patch_num_1]
        confidence_scores = similarity_flat.max(dim=-1)[0]  # corresponding confidence score for each index
        
        position_2 = (
            torch.stack(
                (
                    index_num_2 // feat2.shape[2],
                    index_num_2 % feat2.shape[2]
                ),
                dim = -1
            ) + 0.5
        ) * patch_size
        position_2 = position_2.to('cuda')
        # position_2 is the pixel-center coordinate of the most similar patch in Image 2 corresponding to the i-th patch of Image 1
        
        if pair == 0:
            # foreground patches mask in the left image
            patches_left_fg_selection = (seva_mask_tensor.view(-1) > MASK_FG_THRESHOLD).to('cuda')
            # foreground patches mask in the right image which is resorted by the index_num_2
            patches_right_fg_selection = (input_mask_tensor.to('cuda').view(-1)[index_num_2] > MASK_FG_THRESHOLD)
        else: 
            # foreground patches mask in the left image
            patches_left_fg_selection = (input_mask_tensor.view(-1) > MASK_FG_THRESHOLD).to('cuda')
            # foreground patches mask in the right image which is resorted by the index_num_2
            patches_right_fg_selection = (seva_mask_tensor.to('cuda').view(-1)[index_num_2] > MASK_FG_THRESHOLD)
        
        # only keep the patches that are both foreground in the left and right images
        patches_fg_selection = (patches_left_fg_selection * patches_right_fg_selection)

        # select (row, col) coordinates for the matched foreground patches
        position_1_fg = position_1[patches_fg_selection, :]
        position_2_fg = position_2[patches_fg_selection, :]
        confidence_fg = confidence_scores[patches_fg_selection]  # confidence score for the matched foreground patches
    
    scale_left = scaler[0]
    scale_right = scaler[1]
    
    # note: compute_distances_l2 returns the square of the distance, so the threshold also needs to be squared
    indices_to_exclude, indices_to_keep = stratify_points(position_1_fg * scale_left, STRATIFY_DISTANCE_THRESHOLD**2) 
    
    if len(indices_to_keep) == 0:
        return seva_img_name, float('inf')
    
    # sparse_points_left: (N, 2), N: the number of sparse points, each point is (row, col)
    sparse_points_left = position_1_fg[indices_to_keep, :].cpu().numpy()
    sparse_points_right = position_2_fg[indices_to_keep, :].cpu().numpy()
    sparse_confidence = confidence_fg[indices_to_keep].cpu().numpy()  # sparse_confidence: (N,), N is the number of sparse points

    
    if pair == 0:
        img_left = seva_img
        img_right = input_img
    else:
        img_left = input_img
        img_right = seva_img
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_left)
    ax1.set_axis_off()
    ax2 = fig.add_subplot(122)
    ax2.imshow(img_right)
    ax2.set_axis_off()
    # use colormap
    cmap = cm.get_cmap("RdYlBu")  # red-yellow-blue, warm to cold


    sorted_indices = np.argsort(sparse_confidence)[::-1]
    filtered_indices = sorted_indices[:select_points]
    if len(filtered_indices) < select_points:
        print(f" === NOTE ===: filtered_indices: {len(filtered_indices)}, which is less than select_points: {select_points}")
    top_confidence = filtered_indices.tolist()

    # sorted_indices = np.argsort(sparse_confidence)[::-1]  
    # top_confidence = sorted_indices[:select_points].tolist()
    top_points_left = sparse_points_left[top_confidence]
    top_points_right = sparse_points_right[top_confidence]


    # ===================== visulization =======================
# normalize the confidence score for the colormap
    if visulization_folder is not None and input_img_name is not None and index is not None:
        conf_values = sparse_confidence[top_confidence]
        conf_norm = (conf_values - conf_values.min()) / (conf_values.max() - conf_values.min() + 1e-8)

        cor_x_left, cor_y_left, cor_x_right, cor_y_right = [], [], [], []
        
        for j, (col_left, row_left), (col_right, row_right), norm_val in zip(
            top_confidence, top_points_left, top_points_right, conf_norm
        ):
            cor_y_left.append(round(float(row_left * scale_left), 4))
            cor_x_left.append(round(float(col_left * scale_left), 4))
            cor_y_right.append(round(float(row_right * scale_right), 4))
            cor_x_right.append(round(float(col_right * scale_right), 4))
            color = cmap(norm_val)  # RGBA tuple
            con = ConnectionPatch(
                xyA=(row_left * scale_left, col_left * scale_left),
                xyB=(row_right * scale_right, col_right * scale_right),
                coordsA="data",
                coordsB="data",
                axesA=ax1,
                axesB=ax2,
                color=color,
                linewidth=1.3,
                alpha=1.0
            )
            ax2.add_artist(con)

        
        plt.savefig(os.path.join(visulization_folder, f"{input_img_name}_{index}.png"), dpi=300, bbox_inches='tight')
        plt.close()



    # ===================== visulization =======================

    # geo_error_sum = 0.0
    # for (col_left, row_left), (col_right, row_right) in zip(
    #     top_points_left, top_points_right
    # ):
    #     geo_error_sum += np.sqrt((col_left - col_right)**2 + (row_left - row_right)**2)
    
    
    L = np.array([[c, r] for (c, r) in top_points_left], dtype=np.float32)
    R = np.array([[c, r] for (c, r) in top_points_right], dtype=np.float32)
    # cL = np.median(L, axis=0)
    # cR = np.median(R, axis=0)
    cL = L.mean(axis=0)
    cR = R.mean(axis=0)

    Lc = L - cL
    Rc = R - cR

    diff = Lc - Rc
    d = np.sqrt((diff**2).sum(axis=1) + 1e-8)
    
    return seva_img_name, float(d.mean())
    # return seva_img_name, geo_error_sum



def eval_j(cache, j, pair, seva_img_list, seva_mask_list, input_img, input_mask_tensor, select_points, patch_size, scaler, 
                      img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name, index):
    if j not in cache:
        cache[j] = compute_geo_error(j, pair, seva_img_list, seva_mask_list, input_img, input_mask_tensor, select_points, patch_size, scaler, 
                                      img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name, index)
    return cache[j][1]




if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(42)
    select_points = 60
    print(f"select_points: {select_points}")
    num_patches = 60
    sd_model = sd_aug = extractor_vit = None
    seva_folder = args.seva_folder
    sample_folder = "samples-rgb"
    seva_img_folders = [
        f for f in os.listdir(seva_folder) if not f.startswith('.') and os.path.isdir(os.path.join(seva_folder, f))
    ]
    seva_img_folders = sorted(seva_img_folders, key=sort_key)
    if args.start_end_idx is not None:
        start, end = [s.strip().split(".")[0] for s in args.start_end_idx.split(",")]
        start_pair = parse_pair(start)   # (0,0)
        end_pair   = parse_pair(end)     # (10,0)
        seva_img_folders = [
            f for f in seva_img_folders
            if start_pair <= parse_pair(f) <= end_pair
        ]
        
    folder_tuple = [parse_pair(f) for f in seva_img_folders]
    

  
    
    mask_folder = os.path.join(BASE_DIR, "..", "..", "mid_output", "foreground_images")
    input_mask_folder = os.path.join(mask_folder, "input_images")
    seva_mask_folder = os.path.join(mask_folder, "seva_images")
    seva_mask_folder_dir = os.listdir(seva_mask_folder)
    seva_mask_folder_dir = sorted(seva_mask_folder_dir, key=sort_key)
    if args.start_end_idx is not None:
        start, end = [s.strip().split(".")[0] for s in args.start_end_idx.split(",")]
        start_pair = parse_pair(start)   # (0,0)
        end_pair   = parse_pair(end)     # (10,0)
        seva_mask_folder_dir = [
            f for f in seva_mask_folder_dir
            if start_pair <= parse_pair(f) <= end_pair
        ]
  
    
    with open(args.caption_path, "rb") as f:
        img_dict = json.load(f)
    img_dict_list = list(img_dict.items())
    images_path = []
    for path_, info in img_dict_list:
        images_path.append(path_)
    base_names = [os.path.basename(f) for f in images_path]
    if args.start_end_idx is not None:
        start, end = [s for s in args.start_end_idx.split(",")]
        start_i = base_names.index(start)
        end_i = base_names.index(end)
        images_path = images_path[start_i : end_i + 1]
    
    input_masks_path = [
        f for f in os.listdir(input_mask_folder)
        if not f.startswith('.')
    ]
    input_masks_path = sorted(input_masks_path, key=sort_key)
    if args.start_end_idx is not None:
        start, end = [s for s in args.start_end_idx.split(",")]
        start_i = input_masks_path.index(start)
        end_i = input_masks_path.index(end)
        input_masks_path = input_masks_path[start_i : end_i + 1]
    # print("显存总量:", torch.cuda.get_device_properties("cuda").total_memory / 1024**3, "GB")
    # print("已分配:", torch.cuda.memory_allocated("cuda") / 1024**3, "GB")
    # print("已缓存(保留):", torch.cuda.memory_reserved("cuda") / 1024**3, "GB")
    
    i = 0
    j = 0
    temp = None
    while i < len(seva_img_folders):
        if i+1 < len(seva_img_folders) and folder_tuple[i+1] == folder_tuple[i]:
            temp = seva_img_folders[i+1]
            seva_img_folders[i+1] = seva_img_folders[i+2]
            seva_img_folders[i+2] = temp
            temp = seva_mask_folder_dir[i+1]
            seva_mask_folder_dir[i+1] = seva_mask_folder_dir[i+2]
            seva_mask_folder_dir[i+2] = temp
            images_path[j+1:j+1] = [images_path[j-1], images_path[j]]
            input_masks_path[j+1:j+1] = [input_masks_path[j-1], input_masks_path[j]]
            i += 4
            j += 3
        else:
            i += 1
            j += 1
    
    seva_images_path_list = []
    for i in range(len(seva_img_folders)):
        seva_img_folder = os.path.join(seva_folder, seva_img_folders[i], sample_folder)
        seva_img_list = [
            f for f in os.listdir(seva_img_folder) if not f.startswith('.') and os.path.isfile(os.path.join(seva_img_folder, f))
        ]
        seva_img_list = sorted(seva_img_list, key=sort_key)
        seva_images_path = [os.path.join(seva_img_folder, img) for img in seva_img_list]
        seva_images_path_list.append(seva_images_path)
    seva_mask_path_list = []
    for i in range(len(seva_mask_folder_dir)):
        temp_folder = os.path.join(seva_mask_folder, seva_mask_folder_dir[i])
        temp_mask_path = [
            f for f in os.listdir(temp_folder) if not f.startswith('.') and os.path.isfile(os.path.join(temp_folder, f))
        ]
        temp_mask_path = sorted(temp_mask_path, key=sort_key)
        seva_mask_path_list.append([os.path.join(temp_folder, mask) for mask in temp_mask_path])
        
        
    aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load(f'{BASE_DIR}/results_spair/best_856.PTH'))
    
    # loading model may take a while 
    sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])

    
        
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    
    
    img_size = 960
    assert len(images_path) % 2 == 0, "The number of images must be even"
    print(f"images_path: {images_path}")
    print(f"input_masks_path: {input_masks_path}")
    print(f"seva_images_path_list length: {len(seva_images_path_list)}")
    print(f"seva_mask_path_list length: {len(seva_mask_path_list)}")
    assert len(images_path) == len(input_masks_path), "The number of images and input masks must be the same"
    
    patch_size = 16 
    feat_dim = 768
    # quantization filter for the given patch size
    patch_quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
    transform = transforms.Compose([           
                                transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(),                    
                                ])
    scaler = [512 / img_size, 512 / img_size]
    
    
    # for folder in seva_folder_dir:
    #     seva_masks_path = [
    #         f for f in os.listdir(os.path.join(seva_mask_folder, folder))
    #         if not f.startswith('.')
    #     ]
    #     seva_masks_path = sorted(seva_masks_path, key=sort_key)
    
    
    
    folder_name = "sparse_matching"
    json_folder = os.path.join(BASE_DIR, "..", "..", "mid_output", f"{folder_name}_json")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    else:
        for name in os.listdir(json_folder):
            path = os.path.join(json_folder, name)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                
                
    # visulization_folder = os.path.join(BASE_DIR, "..", "..", "mid_output", f"{folder_name}_visualization")
    # if not os.path.exists(visulization_folder):
    #     os.makedirs(visulization_folder)
    # else:
    #     for name in os.listdir(visulization_folder):
    #         path = os.path.join(visulization_folder, name)
    #         if os.path.isfile(path):
    #             os.remove(path)
    #         elif os.path.isdir(path):
    #             shutil.rmtree(path)
    visulization_folder = None
                
                
    assert len(images_path) == len(seva_img_folders), "The number of images and seva folders must be the same"
    assert len(seva_img_folders) == len(seva_mask_folder_dir), "The number of seva img folders and seva mask folders must be the same"
    best_index_list = []
    name_list = []
    for i in range(0, len(images_path), 2):
        input_img_name_1 = seva_img_folders[i]
        input_img_name_2 = seva_img_folders[i+1]
        print(f"========== Processing input_img_name_1: {input_img_name_1}, input_img_name_2: {input_img_name_2} ==========")
        input_mask_name_1 = input_masks_path[i].split('/')[-1].split('.')[0]
        input_mask_name_2 = input_masks_path[i+1].split('/')[-1].split('.')[0]
        # input_img1_path = os.path.join(image_folder, images_path[i]) # path to the source image
        input_img1_path = images_path[i]
        # input_img2_path = os.path.join(image_folder, images_path[i+1]) # path to the target image
        input_img2_path = images_path[i+1]
        input_img1 = resize(Image.open(input_img1_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)
        input_img2 = resize(Image.open(input_img2_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)
        input_mask1_path = os.path.join(input_mask_folder, input_masks_path[i]) # path to the source mask
        input_mask2_path = os.path.join(input_mask_folder, input_masks_path[i+1]) # path to the target mask
        input_mask1 = Image.open(input_mask1_path).convert('L')
        input_mask2 = Image.open(input_mask2_path).convert('L')
        input_mask_tensor_1 = transform(input_mask1)
        input_mask_tensor_2 = transform(input_mask2)
        input_mask_tensor_1 = patch_quant_filter(input_mask_tensor_1)
        input_mask_tensor_2 = patch_quant_filter(input_mask_tensor_2)
        
        seva_img_list_1 = seva_images_path_list[i]
        seva_img_list_2 = seva_images_path_list[i+1]
        seva_mask_list_1 = seva_mask_path_list[i]
        seva_mask_list_2 = seva_mask_path_list[i+1]
        
        
        # # input img 2 & seva img 1
        geo_error_list_1 = []
        geo_error_list_2 = []
        cache = {}
        
        
        L = 0
        R = len(seva_img_list_1) - 1
        # ternary search to find the best match
        while R - L > 3:
            m1 = L + (R - L) // 3
            m2 = R - (R - L) // 3

            f1 = eval_j(cache, m1, 0, seva_img_list_1, seva_mask_list_1, input_img2, input_mask_tensor_2, select_points, patch_size, scaler, 
                        img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_1, m1)
            f2 = eval_j(cache, m2, 0, seva_img_list_1, seva_mask_list_1, input_img2, input_mask_tensor_2, select_points, patch_size, scaler, 
                        img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_1, m2)

            if f1 < f2:
                R = m2 - 1
            else:
                L = m1 + 1

        # brute force search 
        best_j = -1
        best_err = float('inf')

        for j in range(L, R + 1):
            name, err = compute_geo_error(j, 0, seva_img_list_1, seva_mask_list_1, input_img2, input_mask_tensor_2, select_points, patch_size, scaler, 
                                          img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_1, j)
            if err < best_err:
                best_err = err
                best_j = j
        best_index_1 = best_j

        print(f"========== Best index_1: {best_index_1} ==========")
        
        
        # input img 1 & seva img 2
        cache.clear()
        L = 0
        R = len(seva_img_list_2) - 1
        while R - L > 3:
            m1 = L + (R - L) // 3
            m2 = R - (R - L) // 3
            f1 = eval_j(cache, m1, 1, seva_img_list_2, seva_mask_list_2, input_img1, input_mask_tensor_1, select_points, patch_size, scaler, 
                        img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_2, m1)
            f2 = eval_j(cache, m2, 1, seva_img_list_2, seva_mask_list_2, input_img1, input_mask_tensor_1, select_points, patch_size, scaler, 
                        img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_2, m2)

            if f1 < f2:
                R = m2 - 1
            else:
                L = m1 + 1

        best_j = -1
        best_err = float('inf')

        for j in range(L, R + 1):
            name, err = compute_geo_error(j, 1, seva_img_list_2, seva_mask_list_2, input_img1, input_mask_tensor_1, select_points, patch_size, scaler, 
                                          img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_2, j)
            if err < best_err:
                best_err = err
                best_j = j

        best_index_2 = best_j
        print(f"========== Best index_2: {best_index_2} ==========")
        name_list.append(input_img_name_1)
        name_list.append(input_img_name_2)
        best_index_list.append(int(best_index_1))
        best_index_list.append(int(best_index_2))
        

        # another search method
        # first search 10, 20, 30, .... 70 to find best matching index
        # first_best_index_1 = 10
        # first_best_index_2 = 10
        # err_1 = float('inf')
        # err_2 = float('inf')
        # for j in range(10, 79, 10):
        #     _ , err = compute_geo_error(j, 0, seva_img_list_1, seva_mask_list_1, input_img2, input_mask_tensor_2, select_points, patch_size, scaler, 
        #                               img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_1, j)
        #     if err < err_1:
        #         err_1 = err
        #         first_best_index_1 = j
        #     _ , err = compute_geo_error(j, 1, seva_img_list_2, seva_mask_list_2, input_img1, input_mask_tensor_1, select_points, patch_size, scaler, 
        #                               img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_2, j)
        #     if err < err_2:
        #         err_2 = err
        #         first_best_index_2 = j
        # # then search [x-9, x+9]
        # best_index_1 = first_best_index_1
        # best_index_2 = first_best_index_2
        # for j in range(first_best_index_1 - 9, first_best_index_1 + 10):
        #     if j == first_best_index_1:
        #         continue
        #     _ , err = compute_geo_error(j, 0, seva_img_list_1, seva_mask_list_1, input_img2, input_mask_tensor_2, select_points, patch_size, scaler, 
        #                               img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_1, j)
        #     if err < err_1:
        #         err_1 = err
        #         best_index_1 = j
        # for j in range(first_best_index_2 - 9, first_best_index_2 + 10):
        #     if j == first_best_index_2:
        #         continue
        #     _ , err = compute_geo_error(j, 1, seva_img_list_2, seva_mask_list_2, input_img1, input_mask_tensor_1, select_points, patch_size, scaler, 
        #                               img_size, transform, patch_quant_filter, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, feat_dim, visulization_folder, input_img_name_2, j)
        #     if err < err_2:
        #         err_2 = err
        #         best_index_2 = j
        # print(f"========== Best index_1: {best_index_1} ==========")
        # print(f"========== Best index_2: {best_index_2} ==========")
        # name_list.append(input_img_name_1)
        # name_list.append(input_img_name_2)
        # best_index_list.append(int(best_index_1))
        # best_index_list.append(int(best_index_2))


    search_dict = {}
    for i in range(len(name_list)):
        search_dict[name_list[i]] = best_index_list[i]
    with open(os.path.join(json_folder, "search.json"), "w") as f:
        json.dump(search_dict, f)
        
            
        