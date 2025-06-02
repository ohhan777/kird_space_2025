# convert_to_dota.py
import os
import json
import glob
import argparse
import sys

# 클래스 매핑 (0-based index) to category names
class_names = {
    0: "motorboat",
    1: "sailboat",
    2: "tugboat",
    3: "barge",
    4: "fishing boat",
    5: "ferry",
    6: "container ship",
    7: "oil tanker",
    8: "drill ship",
    9: "warship",
    10: "fighter aircraft",
    11: "large military aircraft",
    12: "small civilian aircraft",
    13: "large civilian aircraft",
    14: "helicopter",
    15: "small vehicle",
    16: "truck",
    17: "bus",
    18: "train",
    19: "container",
    20: "container group",
    21: "crane",
    22: "bridge",
    23: "dam",
    24: "storage tank",
    25: "sports field",
    26: "stadium",
    27: "swimming pool",
    28: "roundabout",
    29: "helipad",
    30: "wind turbine",
    31: "aquaculture farm",
    32: "marine research station",
}

def convert_obb_to_hbb(coords):
    """픽셀 좌표 4개 OBB (x1,y1...x4,y4) -> HBB (x_min,y_min,x_max,y_max)"""
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def process_file(json_path, output_path, img_w, img_h, bb_type):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_f:
        for feat in data.get('features', []):
            props = feat.get('properties', {})
            raw = props.get('object_imcoords', '')
            if not raw:
                continue
            vals = [float(x) for x in raw.replace(' ', '').split(',')]
            coords = []
            for i in range(0, len(vals), 2):
                x = int(vals[i] + 0.5)
                y = int(vals[i+1] + 0.5)
                # Clamp 0~image_size-1
                x = min(img_w - 1, max(0, x))
                y = min(img_h - 1, max(0, y))
                coords.extend([x, y])

            class_id = int(props.get('type_id', 1)) - 1
            class_name = class_names.get(class_id, 'unknown').replace(' ', '_')

            if bb_type == 'hbb':
                xmin, ymin, xmax, ymax = convert_obb_to_hbb(coords)
                line = f"{xmin} {ymin} {xmax} {ymax} {class_name} 0\n"  # write in integer
            else:
                coord_str = ' '.join(f"{c}" for c in coords)  # write in integer
                line = f"{coord_str} {class_name} 0\n"

            out_f.write(line)


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON label files to per-image DOTA format .txt files')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory containing input JSON label files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output base directory for DOTA .txt files')
    parser.add_argument('--width', type=int, default=1024,
                        help='Image width for normalization')
    parser.add_argument('--height', type=int, default=1024,
                        help='Image height for normalization')
    parser.add_argument('--output_bb_type', choices=['hbb', 'obb'], default='obb',
                        help='Bounding box type: hbb or obb')
    args = parser.parse_args()

    json_dir = args.json_dir
    out_base = args.output_dir or os.path.join(os.path.dirname(json_dir), 'dota')

    for json_path in glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True):
        rel = os.path.relpath(json_path, json_dir)
        txt_name = os.path.splitext(rel)[0] + '.txt'
        out_path = os.path.join(out_base, txt_name)
        process_file(json_path, out_path, args.width, args.height, args.output_bb_type)
        print(f"Converted: {json_path} -> {out_path}")

if __name__ == '__main__':
    main()
