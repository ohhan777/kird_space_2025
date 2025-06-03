import os
import cv2
import numpy as np
import torch

def make_color_label(label):
    class_colors = [        # RGB color for each class
        [0, 0, 0],          # 0: background
        [226, 124, 144],    # 1: motorway
        [251, 192, 172],    # 2: trunk
        [253, 215, 161],    # 3: primary
        [246, 250, 187],    # 4: secondary
        [255, 255, 255],    # 5: tertiary
        [75, 238, 49],      # 6: path
        [173, 173, 173],    # 7: under construction
        [170, 85, 255],     # 8: train guideway
        [120, 232, 234]     # 9: airplay runway
    ]
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3) shape
    for i, class_color in enumerate(class_colors):
        color_label[label == i] = class_color
    return color_label


def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    # if img is a tensor, convert to a numpy array
    if torch.is_tensor(img):  # input: (3, H, W) tensor, RGB, 0-1 normalized
        img = img.mul(255.0).clamp(0.0, 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    # if img is a numpy array, no further processing. input: (H, W, 3), RGB, 0-255

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    if label is not None:
        # if label_img is tensor, convert to numpy array
        if torch.is_tensor(label):
            label = label.cpu().numpy().astype(np.uint8)
            
        color_label = make_color_label(label)
        color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2
        
        img = cv2.addWeighted(img, 1.0, color_label, alpha, 0)  # overlays image and label

    # save image
    cv2.imwrite(save_file, img)

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    img_file = 'data/kari_roads_mini/train/images/BLD11166_PS3_K3A_NIA0390.png'
    label_file = img_file.replace('images', 'png_labels')

    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    
    color_label = make_color_label(label)
    color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs/label1.png', color_label)
    
    plot_image(img, save_file='outputs/image1.png')
    plot_image(img, label, save_file='outputs/image+label1.png')