import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
from DRL.actor import *
from Renderer.stroke_gen import *
# from Renderer.model import *
import pandas as pd
from DRL.actor import ResNet
from Renderer.model import FCN
import matplotlib.pyplot as plt
from test_utils import *
from collections import OrderedDict

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse input arguments
parser = argparse.ArgumentParser(description='Paint canvases using semantic guidance')
parser.add_argument('--max_eps_len', default=50, type=int, help='max length for episode')
parser.add_argument('--actor', default='pretrained_models/actor_semantic_guidance.pkl', type=str, help='actor model')
parser.add_argument('--use_baseline', action='store_true', help='use baseline model instead of semantic guidance')
parser.add_argument('--renderer', default='../data/renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='../image/test.png', type=str, help='test image')
args = parser.parse_args()

# width of canvas used
width = 128
# time image
n_batch = 1
T = torch.ones([n_batch, 1, width, width], dtype=torch.float32).to(device)
coord = torch.zeros([n_batch, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[:, 0, i, j] = i / (width - 1.)
        coord[:, 1, i, j] = j / (width - 1.)
coord = coord.to(device)  # Coordconv

# define, load actor and set to eval mode
bundle_size = 5
use_gbp, use_bilevel, use_neural_alignment = False, False, False
if not args.use_baseline:
    use_gbp, use_bilevel, use_neural_alignment = True, True, True
    bundle_size = int(np.ceil(bundle_size / 2))
actor = ResNet(9 + use_gbp + use_bilevel + 2 * use_neural_alignment, 18, 13 * bundle_size * (1 + use_bilevel))

# load trained actor model
state_dict = torch.load(args.actor, map_location={'cuda:0': 'cpu'})
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # k[7:] # remove `module.`
    new_state_dict[name] = v
actor.load_state_dict(new_state_dict)
actor = actor.to(device).eval()

if not args.use_baseline:
    # load object localization and semantic segmentation model
    seg_model = get_model_instance_segmentation(num_classes=2)
    seg_model.load_state_dict(torch.load('../data/birds_obj_seg.pkl', map_location={'cuda:0': 'cpu'}))
    for param in seg_model.parameters():
        param.requires_grad = False
    seg_model.to(device)
    seg_model.eval()

    # load expert model for getting guided backpropagation maps
    expert_model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                                  **{'topN': 6, 'device': 'cpu', 'num_classes': 200})
    expert_model.eval()
    GBP = GuidedBackprop(expert_model.pretrained_model)

    # get segmentation, bbox and guided backpropagation map predictions
    img = Image.open(args.img)  # cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_ = transform_test(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = seg_model(img_)
    bbox_pred = prediction[0]['boxes'][0]
    x, y, w, h = bbox_pred.detach().cpu().numpy().astype(int)
    seg_mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
    classgt_mask = seg_mask.copy()

    # get guided backpropagation maps from the expert model
    img = cv2.cvtColor(cv2.imread(args.img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W, C = img.shape
    img = img[y:h, x:w].astype(np.uint8)
    seg_mask = seg_mask.astype(float) / 255.
    seg_mask = seg_mask[y:h, x: w]

    img = img.astype(float) * np.expand_dims(seg_mask, -1)
    img = img.astype(np.uint8)

    scaled_img = gbp_transform(img)
    torch_images = scaled_img.unsqueeze(0)

    guided_grads = GBP.generate_gradients(torch_images)  # .transpose(1,2,0)
    gbp = convert_to_grayscale(guided_grads)[0]
    gbp = cv2.resize(gbp, (w - x, h - y))
    gbp = (255 * gbp).astype(np.uint8)

    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    gbp = gbp * seg_mask

    gbp_global = np.zeros((H, W)).astype(np.uint8)
    w, h = min(w, W), min(h, H)
    gbp_global[y:h, x:w] = gbp

    # get grid for spatial transformer network
    w, h = w - x, h - y
    x, y, w, h = x / W, y / H, w / W, h / H
    Affine_Mat_w = [w, 0, (2 * x + w - 1)]
    Affine_Mat_h = [0, h, (2 * y + h - 1)]
    M = np.c_[Affine_Mat_w, Affine_Mat_h].T
    M = torch.tensor(M).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(M, (1, 3, 128, 128))  # (1,128,128,2)
    grid = (grid + 1) / 2  # scale between 0,1
    grid = torch.tensor(grid * 255, dtype=torch.uint8).permute(0, 3, 1, 2)
    grid = grid.to(device).float() / 255.

    # load segmentation image
    classgt_img = cv2.resize(classgt_mask, (128, 128))
    classgt_img = torch.tensor(classgt_img, dtype=torch.uint8)
    classgt_img = classgt_img.unsqueeze(0).unsqueeze(0).to(device).float() / 255.

    # load guided backpropagation map
    gbp_global = cv2.resize(gbp_global, (128, 128))
    gbp_img = torch.tensor(gbp_global, dtype=torch.uint8)
    gbp_img = gbp_img.unsqueeze(0).unsqueeze(0).to(device).float() / 255.

# load image
image = cv2.cvtColor(cv2.imread(args.img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)
img = image.unsqueeze(0).to(device).float() / 255.

# initialize empty canvas
canvas = torch.zeros([n_batch, 3, width, width]).to(device)
max_eps_len = args.max_eps_len
# initialize canvas frames for generating video
canvas_frames = []
canvas_ = canvas.permute(0, 2, 3, 1).cpu().numpy()
canvas_frames.append(canvas_[0])

with torch.no_grad():
    # generate canvas episode
    for i in range(max_eps_len):
        stroke_type = 'both'  # 'fg' if i > 1 else 'both'
        stepnum = T * i / max_eps_len
        if args.use_baseline:
            actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
            canvas, _, frames = decode(actions, canvas, get_frames=True, bundle_size=bundle_size)
        else:
            actions = actor(torch.cat([canvas, img, stepnum, coord, gbp_img, classgt_img, grid], 1))
            canvas, _, frames = decode_parallel(actions, canvas, seg_mask=classgt_img, stroke_type=stroke_type,
                                                get_frames=True, bundle_size=bundle_size)
        canvas_frames += frames

# get final painted canvas
canvas_ = canvas.permute(0, 2, 3, 1).cpu().numpy()
canvas_ = np.array(255 * canvas_[0]).astype(np.uint8)
# H, W, C = cv2.imread(args.img, cv2.IMREAD_COLOR).shape
H, W = 250, 250
painted_canvas = cv2.cvtColor(cv2.resize(canvas_, (W, H)), cv2.COLOR_RGB2BGR)

# save generated canvas
save_img_name = "../output/{}_painted_{}".format('baseline' if args.use_baseline else 'sg', os.path.basename(args.img))
print("\nSaving painted canvas to {}\n".format(save_img_name))
cv2.imwrite(save_img_name, painted_canvas)

# generate and save video for the painting episode
video_name = '../video/{}_{}.mp4'.format('baseline' if args.use_baseline else 'sg', os.path.basename(args.img)[:-4])
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (W, H))
for image in canvas_frames:
    image = np.array(255 * image, dtype=np.uint8)
    image = cv2.resize(image, (W, H))
    video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
video.release()
print("\nSaving painting video to {}\n".format(video_name))
