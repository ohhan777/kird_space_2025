import cv2
import numpy as np
import os

def make_color_label(label):
    class_colors = [        # RGB color for each class
        [0, 0, 0],         # 0: background
        [226, 124, 144],   # 1: motorway
        [251, 192, 172],   # 2: trunk
        [253, 215, 161],   # 3: primary
        [246, 250, 187],   # 4: secondary
        [255, 255, 255],   # 5: tertiary
        [75, 238, 49],     # 6: path
        [173, 173, 173],   # 7: under construction
        [170, 85, 255],    # 8: train guideway
        [120, 232, 234]    # 9: airplay runway
    ]
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3) shape
    for i, class_color in enumerate(class_colors):
        color_label[label == i] = class_color
    return color_label

def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    color_label = make_color_label(label)
    color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2
    
    img = cv2.addWeighted(img, 1.0, color_label, alpha, 0)  # overlays image and label
    # save image
    cv2.imwrite(save_file, img)


img_file = 'data/kari_roads_mini/train/images/BLD11166_PS3_K3A_NIA0390.png'
label_file = img_file.replace('images', 'png_labels')
img = cv2.imread(img_file)
label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
output_file = os.path.join('outputs', os.path.basename(img_file).replace('.png', '_overlay.png'))
plot_image(img, label, save_file=output_file)
