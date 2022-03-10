import torch
import sys
import os
# sys.path.append('./')
from matplotlib import cm
import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchvision.utils import save_image
import os

class_names = ['OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak', 'Flower']

bn_class_names = ['OK', 'NoOK']

header_names = ['filename'] + class_names 

class_label_to_name = {0: 'ok', 1: 'qishihuangdong', 2 : 'sipei', 3: 'kongliao', 4: 'wuqishi', 5: 'liewen', 6: 'ruopei', 7: 'huake'}
name_to_class_label = {'ok': 0 , 'qishihuangdong' : 1, 'sipei' : 1, 'kongliao' : 1, 'wuqishi' : 1, 'liewen' : 1, 'ruopei' : 1}

def extract_activation_map(cam_extractor, images, preds):
    activation_map = cam_extractor(preds.argmax(dim=1).tolist(), preds)[0]
    activation_map = torch.nn.functional.interpolate(activation_map.unsqueeze(1), 
                                                     size = images.size()[2:], 
                                                     mode ='bicubic',
                                                     align_corners=True)
    # [b, 1, 3, h, w]
    return activation_map.to(images.device)


def generate_heatmaps(activation_maps, colormap='jet'):
    """generate the corresponding heatmaps using activation maps

    Args:
        activation_maps (_type_): [b, n, h, w], b is the batch size, n is the total layers, h and w are the image spatical size
        mask (_type_): _description_
        colormap (str, optional): _description_. Defaults to 'jet'.

    Returns:
        _type_: _description_
    """
    b, n, h, w = activation_maps.size()
    activation_maps = activation_maps.reshape(b * n, h, w)
    heat_maps = []
    for mask in activation_maps:
        cmap = cm.get_cmap(colormap)
        # generate a heat map, [h, w, 3]
        heat_map = torch.from_numpy(cmap((mask ** 2).detach().cpu().numpy())[:, :, :3]).to(mask.device)
        heat_maps.append(heat_map)
        # print(heat_map.size())
        # [b, n, h, w, 3] -> [b, n, 3, h, w]
    return torch.stack(heat_maps).reshape(b, n, h, w, -1).permute(0, 1, 4, 2, 3)
    
def overlay(images, heat_maps, alpha = 0.5):
    """_summary_

    Args:
        images (_type_): [b, 1, 3, h, w]
        heat_maps (_type_): [b, n, 3, h, w] 
        alpha (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    return images * alpha + (1 - alpha) * heat_maps

    
def render_label(image, label, pred):
    image = cv2.putText(image, f'Label: {label}', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Pred: {pred}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('Classification Result',image)
    # cv2.waitKey(1)
    return image

def render_labels(images, labels, preds):
    """

    Args:
        images (_type_): [b, 1, 3, h, w]
        labels (_type_): _description_
        preds (_type_): _description_

    Returns:
        _type_: _description_
    """
    cpu_images = images.detach().cpu().squeeze()
    # one-hot to class index
    preds = preds.argmax(dim=1).tolist()
    labels = labels.argmax(dim=1).tolist()
    new_images = []
    for image, label, pred in zip(cpu_images, labels, preds):
        image = np.array(to_pil_image(image))
        # class index to class name
        label = class_label_to_name[label]
        pred = class_label_to_name[pred]
        image = render_label(image, label, pred)
        new_images.append(to_tensor(image))
        # [b, 1, 3, h, w]
    return torch.stack(new_images).unsqueeze(1)


if __name__ == '__main__':
    images = Image.open('/data/lxd/datasets/2022-03-02-Eggs/OK/000445139_Egg6_(ok)_L_0_cam5.bmp')
    images = to_tensor(images).reshape(1, 3, 700, 600)
    images = render_labels(images, torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
    save_image(images, os.path.join('test_results', 'test_render_label.jpeg'))
    

        
        
        