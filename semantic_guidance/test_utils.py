import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms, utils
import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from Renderer.model import FCN
from torch.nn import ReLU

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# renderer model
Decoder = FCN()
Decoder.load_state_dict(torch.load('../data/renderer.pkl'))
Decoder = Decoder.to(device).eval()

def decode_parallel(x, canvas, seg_mask=None, mask=False, stroke_type='both', get_frames=False, bundle_size=3):
    # print (seg_mask.shape)

    bg, fg = True, True

    if stroke_type == 'fg':
        bg = False
    if stroke_type == 'bg':
        fg = False

    canvas_frames = []
    if get_frames:
        if bg:
            canvas, _, frames = decode(x[:, :13 * bundle_size], canvas, mask, 1 - seg_mask, get_frames=get_frames,bundle_size=bundle_size)
            canvas_frames += frames
        if fg:
            canvas, _, frames = decode(x[:, 13 * bundle_size:], canvas, mask, seg_mask, get_frames=get_frames,bundle_size=bundle_size)
            canvas_frames += frames
        return canvas, _, canvas_frames

    if bg:
        canvas, _ = decode(x[:, :13 * bundle_size], canvas, mask, 1 - seg_mask, get_frames=get_frames,bundle_size=bundle_size)
    if fg:
        canvas, _ = decode(x[:, 13 * bundle_size:], canvas, mask, seg_mask, get_frames=get_frames,bundle_size=bundle_size)
    return canvas, _


def decode(x, canvas, mask=False, seg_mask=None, get_frames=False, bundle_size=5):  # b * (10 + 3)
    """
    Update canvas given stroke parameters x
    """
    # 13 stroke parameters (10 position and 3 RGB color)
    x = x.contiguous().view(-1, 10 + 3)

    # get stroke on an empty canvas given 10 positional parameters
    stroke = 1 - Decoder(x[:, :10])

    stroke = stroke.view(-1, 128, 128, 1)

    # add color to the stroke
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)

    stroke = stroke.view(-1, bundle_size, 1, 128, 128)
    color_stroke = color_stroke.view(-1, bundle_size, 3, 128, 128)

    canvas_frames = []
    for i in range(bundle_size):
        if seg_mask is not None:
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i] * seg_mask
        else:
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        canvas_ = canvas.permute(0, 2, 3, 1).cpu().numpy()
        canvas_frames.append(canvas_[0])

    # also return stroke mask if required
    stroke_mask = None
    if mask:
        stroke_mask = (stroke != 0).float()  # -1, bundle_size, 1, width, width

    if get_frames:
        return canvas, stroke_mask, canvas_frames
    return canvas, stroke_mask


gbp_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((440, 440)),
    transforms.CenterCrop((440, 440)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # print (module,grad_in,grad_out)
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image):
        input_image.requires_grad = True
        # Forward pass
        model_output = self.model(input_image)[0]
        target_class = torch.argmax(model_output).item()

        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # print (one_hot_output)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
