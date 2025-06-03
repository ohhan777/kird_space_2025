import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils.plots import plot_image
import argparse
from torchvision import models

def predict(opt):   
    # Model initialization
    model = models.segmentation.deeplabv3_resnet101(num_classes=10)

    # GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Load weights
    if not os.path.exists(opt.weight):
        raise FileNotFoundError(f"Model weight file not found: {opt.weight}")
    checkpoint = torch.load(opt.weight, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Input processing
    if not os.path.exists(opt.input):
        raise FileNotFoundError(f"Input image not found: {opt.input}")
    img = cv2.imread(opt.input)
    if img is None:
        raise ValueError(f"Failed to read image: {opt.input}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # Convert BGR to RGB
    img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0  # Normalize to [0,1]
    imgs = img.unsqueeze(0)  # Add batch dimension (1, 3, H, W)

    # Create output directory
    os.makedirs(opt.output, exist_ok=True)
    
    print('Predicting...')
    model.eval()
    imgs = imgs.to(device)
    with torch.no_grad():
        preds = model(imgs)['out']  # (1, C, H, W)
        preds = torch.argmax(preds, dim=1)  # (1, H, W)
        save_file = os.path.join(opt.output, os.path.basename(opt.input).replace('.png', '_pred.png'))
        plot_image(imgs[0], preds[0], save_file)     
    print('Done')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--output', '-o', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--weight', '-w', default='weights/ohhan_best.pth', help='Weight file path')
    opt = parser.parse_args()

    predict(opt)