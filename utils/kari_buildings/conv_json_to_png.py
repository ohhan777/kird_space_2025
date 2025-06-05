import os
import json
import glob
import argparse
import numpy as np
import cv2

def process_file(json_path, output_path, img_w, img_h):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a single blank image for all features
    png_label = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        # find coordinates from "building_imcoords" or "road_imcoords"
        raw = props.get('building_imcoords', '') if props.get('building_imcoords', '') != 'EMPTY' else props.get('road_imcoords', '')
        if raw == 'EMPTY' or raw == '':
            continue
        vals = [round(float(x)) for x in raw.replace(' ', '').split(',')]
        cls_id = int(props.get('type_id', 1))
        cls_id = 1 if cls_id >= 1 else 0
        vals = np.array(vals).reshape(-1, 2)
        png_label = cv2.fillPoly(png_label, [vals], cls_id)
    
    cv2.imwrite(output_path, png_label)


def main():
    p = argparse.ArgumentParser(
        description="Convert JSON labels to PNG label files (per-image.png)")
    p.add_argument('--json_dir', required=True,
                   help='Input directory with JSON label files')
    p.add_argument('--output_dir', default=None,
                   help='Output base directory for PNG label files')
    p.add_argument('--width', type=int, default=1024,
                   help='Image width')
    p.add_argument('--height', type=int, default=1024,
                   help='Image height')
    args = p.parse_args()

    json_dir = os.path.normpath(args.json_dir)  # 마지막에 "/"와 상관없이 작동하도록
    out_base = args.output_dir or os.path.join(os.path.dirname(json_dir), 'png_labels')

    # 순회
    for json_path in glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True):
        rel = os.path.relpath(json_path, json_dir)
        txt_name = os.path.splitext(rel)[0] + '.png'
        out_path = os.path.join(out_base, txt_name)
        process_file(json_path, out_path, args.width, args.height)
        print(f"Converted: {json_path} -> {out_path}")

if __name__ == '__main__':
    main()
