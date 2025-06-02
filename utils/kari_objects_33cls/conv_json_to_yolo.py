import os
import json
import glob
import argparse
import sys

# 클래스 ID (0-based) 에 대응되는 이름 매핑 (공간 구분 없음)
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
    """정규화된 4개 OBB 점(x1,y1...x4,y4)을 (center_x,center_y,width,height)로 변환"""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return [cx, cy, w, h]


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
            # 정규화
            norm = []
            for i in range(0, len(vals), 2):
                x = vals[i] / img_w
                y = vals[i+1] / img_h
                x = min(1.0, max(0.0, x))
                y = min(1.0, max(0.0, y))
                norm.extend([x, y])

            cls_id = int(props.get('type_id', 1)) - 1
            if bb_type == 'hbb':
                cx, cy, w, h = convert_obb_to_hbb(norm)
                line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            else:
                coords_str = ' '.join(f"{v:.6f}" for v in norm)
                line = f"{cls_id} {coords_str}\n"
            out_f.write(line)


def main():
    p = argparse.ArgumentParser(
        description="Convert JSON labels to YOLO format files (per-image .txt)")
    p.add_argument('--json_dir', required=True,
                   help='Input directory with JSON label files')
    p.add_argument('--output_dir', default=None,
                   help='Output base directory for YOLO .txt files')
    p.add_argument('--width', type=int, default=1024,
                   help='Image width for normalization')
    p.add_argument('--height', type=int, default=1024,
                   help='Image height for normalization')
    p.add_argument('--output_bb_type', choices=['hbb','obb'], default='obb',
                   help='Bounding box type: hbb (horizontal) or obb (oriented)')
    args = p.parse_args()

    json_dir = args.json_dir
    out_base = args.output_dir or os.path.join(os.path.dirname(json_dir), 'yolo')

    # 순회
    for json_path in glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True):
        rel = os.path.relpath(json_path, json_dir)
        txt_name = os.path.splitext(rel)[0] + '.txt'
        out_path = os.path.join(out_base, txt_name)
        process_file(json_path, out_path, args.width, args.height, args.output_bb_type)
        print(f"Converted: {json_path} -> {out_path}")

if __name__ == '__main__':
    main()
