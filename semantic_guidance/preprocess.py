import numpy as np
import os, sys
import pandas as pd
import cv2

from PIL import Image
import torch
import torchvision
from torchvision import transforms, utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device: {}".format(device))


# define function for getting combined object localization and semantic segmentation prediction model
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


# our dataset has two classes only - background and foreground
num_classes = 2
# load pretrained object localization and semantic segmentation model
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load('../data/birds_obj_seg.pkl', map_location={'cuda:0': 'cpu'}))
model.to(device)
model.eval()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

expert_model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                              **{'topN': 6, 'device': 'cpu', 'num_classes': 200})
expert_model.eval()

from torch.nn import ReLU

gbp_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((440, 440)),
    transforms.CenterCrop((440, 440)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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


# df_ = pd.read_csv('data/cub200/CUB_200_2011/bounding_boxes.txt', sep=' ', index_col=0,
#                   names=['idx', 'x', 'y', 'w', 'h'])
# df_ = df_.copy()
# data = []

# get input image list
df = pd.read_csv('../data/cub200/CUB_200_2011/images.txt', sep=' ', index_col=0, names=['idx', 'img_names'])
img_names = list(df['img_names'])
img_list = np.array(["../data/cub200/CUB_200_2011/images/{}.jpg".format(img_name[:-4]) for img_name in img_names])

# predicted bounded box data
pred_bbox_data = []
GBP = GuidedBackprop(expert_model.pretrained_model)

# create output directories for storing data
gbp_dir = '../data/cub200/CUB_200_2011/gbp_global/'
segmentations_dir = '../data/cub200/CUB_200_2011/segmentations_pred/'
if not os.path.exists(gbp_dir):
    os.makedirs(gbp_dir)
if not os.path.exists(segmentations_dir):
    os.makedirs(segmentations_dir)

for choose in range(len(img_list)):
    # get image segmentation and bounding box predictions
    img_name = img_list[choose]
    img = Image.open(img_name)  # cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_ = transform_test(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_)
    bbox_pred = prediction[0]['boxes'][0]
    x, y, w, h = bbox_pred.detach().cpu().numpy().astype(int)
    seg_mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    pred_bbox_data.append([x, y, w - x, h - y])
    # df_.iloc[choose]['x'] = x
    # df_.iloc[choose]['y'] = y
    # df_.iloc[choose]['w'] = w - x
    # df_.iloc[choose]['h'] = h - y

    path_list = img_name.split('/')
    path_list[-3] = 'segmentations_pred'
    seg_name = '/'.join(path_list)
    # seg_name = "{}/segmentations_pred2/{}".format(img_name[:24], img_name[32:])
    if not os.path.exists(os.path.dirname(seg_name)):
        os.makedirs(os.path.dirname(seg_name))
    cv2.imwrite(seg_name, seg_mask)

    # get guided backpropagation maps from the expert model
    img = cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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
    gbp = cv2.resize(gbp, (w-x, h-y))
    gbp = (255 * gbp).astype(np.uint8)

    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    gbp = gbp * seg_mask

    gbp_global = np.zeros((H, W)).astype(np.uint8)
    w, h = min(w, W), min(h, H)
    gbp_global[y:h, x:w] = gbp

    path_list = img_name.split('/')
    path_list[-3] = 'gbp_global'
    gbp_name = '/'.join(path_list)
    if not os.path.exists(os.path.dirname(gbp_name)):
        os.makedirs(os.path.dirname(gbp_name))
    cv2.imwrite(gbp_name, gbp_global)

    if choose % 100 == 0:
        print("Processed :{}/{} images ...".format(choose, len(img_list)))

# save bounding box predictions
df_ = pd.DataFrame(data=pred_bbox_data, columns=['x', 'y', 'w', 'h'])
df_.to_csv('../data/cub200/CUB_200_2011/bounding_boxes_pred.txt', sep=' ')
