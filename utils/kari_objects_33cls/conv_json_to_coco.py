import os
import json
import glob
import argparse

# Class mapping (0-based index) to category names
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert JSON label files to a single COCO-format JSON.')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory containing input JSON label files')
    parser.add_argument('--output', type=str, default='train.json',
                        help='Path to output COCO JSON file')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory with corresponding images (optional)')
    parser.add_argument('--image_ext', type=str, default='.png',
                        help='Image file extension (if --image_dir is set)')
    parser.add_argument('--width', type=int, default=1024,
                        help='Width of images in pixels')
    parser.add_argument('--height', type=int, default=1024,
                        help='Height of images in pixels')
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Fill categories
    for idx, name in class_names.items():
        coco["categories"].append({
            "id": idx + 1,
            "name": name.replace(' ', '_'),
            "supercategory": "none"
        })

    ann_id = 1
    img_id = 1

    # Iterate JSON files
    for json_path in sorted(glob.glob(os.path.join(args.json_dir, '*.json'))):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Determine image file name
        if args.image_dir:
            file_base = os.path.splitext(os.path.basename(json_path))[0]
            file_name = file_base + args.image_ext
        else:
            file_name = os.path.basename(json_path)

        # Add image entry
        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": args.width,
            "height": args.height
        })

        # Parse each object in JSON
        for feature in data.get('features', []):
            props = feature['properties']
            # parse polygon coords: object_imcoords as "x0,y0,x1,y1,..."
            coords = [float(x) for x in props['object_imcoords'].split(',')]

            # segmentation: polygon points
            segmentation = [coords]

            # compute bbox: [x_min, y_min, width, height]
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, y_min = min(xs), min(ys)
            width = max(xs) - x_min
            height = max(ys) - y_min
            bbox = [x_min, y_min, width, height]

            # category id (1-based)
            cat_id = int(props['type_id'])

            # area
            area = width * height

            # annotation entry
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    # Write out COCO JSON
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as out_f:
        json.dump(coco, out_f, indent=2)

    print(f"COCO JSON saved to {args.output} (images: {img_id-1}, annotations: {ann_id-1})")


if __name__ == '__main__':
    main()
