import json
import os
import sys
import numpy as np
import glob


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


def get_image_corners(data):
    # 첫 번째 feature의 geometry coordinates를 이미지 모서리로 사용
    if 'features' in data and len(data['features']) > 0:
        first_feature = data['features'][0]
        if 'geometry' in first_feature and 'coordinates' in first_feature['geometry']:
            coords = first_feature['geometry']['coordinates']

            if len(coords) >= 4:
                # coordinates는 [lon, lat, z] 형태이므로 z를 제거하고 [lon, lat]만 사용
                coords_list = []
                for coord in coords:
                    if len(coord) >= 2:
                        coords_list.append([coord[0], coord[1]])  # [lon, lat]
                # print(f"Extracted image corners from JSON: {coords_list}")
                return coords_list  # [좌상단, 우상단, 우하단, 좌하단]
        
        raise ValueError("No valid coordinates found in JSON file geometry")
    

# 픽셀 좌표를 위경도 좌표로 변환 (회전 옵션 포함)
def pixel_to_geo(pixel_coords, image_corners, width=1024, height=1024):
    lon_lt, lat_lt = image_corners[0]  # 좌상단
    lon_rt, lat_rt = image_corners[1]  # 우상단
    lon_rb, lat_rb = image_corners[2]  # 우하단
    lon_lb, lat_lb = image_corners[3]  # 좌하단

    geo_coords = []
    for i in range(0, len(pixel_coords), 2):
        x, y = pixel_coords[i], pixel_coords[i + 1]

        x, y = x, height - y 

        # 보간 비율
        x_ratio = x / width
        y_ratio = y / height

        # 선형 보간 (bilinear interpolation)
        lon_top = lon_lt + x_ratio * (lon_rt - lon_lt)
        lon_bottom = lon_lb + x_ratio * (lon_rb - lon_lb)
        lon = lon_top + y_ratio * (lon_bottom - lon_top)

        lat_top = lat_lt + x_ratio * (lat_rt - lat_lt)
        lat_bottom = lat_lb + x_ratio * (lat_rb - lat_lb)
        lat = lat_top + y_ratio * (lat_bottom - lat_top)

        geo_coords.append([lon, lat])

    # 폴리곤 닫기
    geo_coords.append(geo_coords[0])
    # print(f"Converted geo coordinates for pixel {pixel_coords[:2]}: {geo_coords[:1]}")
    return geo_coords

def convert_obb_to_hbb(pixel_coords):
    """OBB 좌표를 HBB 좌표로 변환"""
    min_x = min(pixel_coords[0::2])
    min_y = min(pixel_coords[1::2])
    max_x = max(pixel_coords[0::2])
    max_y = max(pixel_coords[1::2])
    return [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]

def calculate_orientation(pixel_coords, props):
    """픽셀 좌표와 속성으로부터 orientation 계산"""
    if 'object_angle' not in props:
        angle = np.arctan2(pixel_coords[3] - pixel_coords[1], pixel_coords[2] - pixel_coords[0])
        angle = np.degrees(angle)
    else:
        angle = float(props['object_angle'])
    return angle + 360 if angle < 0 else angle

def create_feature(geo_coords, class_id, pixel_coords, bb_type, props=None):
    """GeoJSON Feature 객체 생성"""
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [geo_coords]
        },
        "properties": {
            "class_id": class_id,
            "class_name": class_names[class_id - 1],
            "pixel_coordinates": pixel_coords
        }
    }
    
    # OBB의 경우 orientation 추가
    if bb_type == "obb" and props:
        orientation = calculate_orientation(pixel_coords, props)
        feature["properties"]["orientation"] = round(orientation, 2)
    
    return feature

def create_geojson_structure():
    """빈 GeoJSON 구조 생성"""
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": []
    }

# GeoJSON 생성
def create_geojson(json_path, output_path, bb_type, image_size=(1024, 1024)):
    try:
        # JSON 파일 읽기
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_corners = get_image_corners(data)
        
        # GeoJSON 구조 초기화
        geojson = create_geojson_structure()
        
        # 각 Feature 처리
        for feature in data['features']:
            props = feature['properties']
            # 픽셀 좌표 파싱
            pixel_coords = [float(x) for x in props['object_imcoords'].replace(' ', '').split(',')]
            
            # HBB의 경우 OBB를 HBB로 변환
            if bb_type == "hbb":
                pixel_coords = convert_obb_to_hbb(pixel_coords)

            # 위경도 좌표 변환
            geo_coords = pixel_to_geo(pixel_coords, image_corners, image_size[0], image_size[1])
            
            # 공통 Feature 구조 생성
            class_id = int(props['type_id'])
            new_feature = create_feature(geo_coords, class_id, pixel_coords, bb_type, props)
            
            geojson['features'].append(new_feature)
        
        # GeoJSON 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        # print(f"GeoJSON file created: {output_path}")
    
    except Exception as e:
        print(f"Error creating GeoJSON: {e}")
        sys.exit(1)

def process_files():
    """모든 JSON 파일을 처리하여 GeoJSON으로 변환"""
    bb_type = "hbb"
    for json_path in glob.glob("val/labels/obb/json/*.json"):
        print(json_path, end="...")
        output_path = json_path.replace(".json", ".geojson").replace("json/", "geojson/").replace("obb/", bb_type + "/")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        
        create_geojson(json_path, output_path, bb_type)
        print("done")

# 메인 실행
if __name__ == "__main__":
    process_files()