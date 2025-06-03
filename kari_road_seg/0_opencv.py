import os
import cv2

def plot_image_file(img_file):
    img = cv2.imread(img_file)
    output_file = os.path.join('outputs', os.path.basename(img_file))
    cv2.imwrite(output_file, img)

def process_image(img_file):
    plot_image_file(img_file)

img_file = 'data/kari_roads_mini/train/images/BLD11166_PS3_K3A_NIA0390.png'
plot_image_file(img_file)
