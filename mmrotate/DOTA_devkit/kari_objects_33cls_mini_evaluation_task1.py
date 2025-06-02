import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial
import glob

def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]
                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath.format(imagename))
    
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([bool(x['difficult']) for x in R])
        det = [False] * len(R)
        npos = npos + sum(difficult == False)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    detection_results = {}
    
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        
        if BBGT.size > 0:
            # Compute overlaps
            # Intersection
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # Union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]

        is_correct = ovmax > ovthresh and not R['difficult'][jmax] if ovmax > -np.inf else False

        if image_ids[d] not in detection_results:
            detection_results[image_ids[d]] = []
        
        detection_results[image_ids[d]].append({
            'bbox': bb.tolist(),
            'class_name': classname,
            'score': confidence[sorted_ind[d]],
            'is_correct': is_correct
        })

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    
    return rec, prec, ap, detection_results

def main():

    detpath = r'work_dirs/Task1_results/Task1_{:s}.txt'
    annopath = r'data/kari_objects_33cls_mini/val/dota/{:s}.txt'
    img_path = r'data/kari_objects_33cls_mini/val/images'
    img_list_file = r'data/kari_objects_33cls_mini_val_image_list.txt'
    import glob
    img_files = glob.glob(os.path.join(img_path, '*.png'))
    img_files = [os.path.basename(img_file).split('.')[0] for img_file in img_files]

    with open(img_list_file, 'w') as f:
        for img_file in img_files:
            f.write(img_file + '\n')

    classnames = ['motorboat', 'sailboat', 'tugboat', 'barge', 'fishing_boat', 'ferry', 'container_ship', 
               'oil_tanker', 'drill_ship', 'warship', 'fighter_aircraft', 'large_military_aircraft', 'small_civilian_aircraft', 
               'large_civilian_aircraft', 'helicopter', 'small_vehicle', 'truck', 'bus', 'train', 'container', 'container_group', 
               'crane', 'bridge', 'dam', 'storage_tank', 'sports_field', 'stadium', 'swimming_pool', 'roundabout', 'helipad', 
               'wind_turbine', 'aquaculture_farm', 'marine_research_station']

    all_detection_results = {}
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap, class_results = voc_eval(detpath,
             annopath,
             img_list_file,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        
        for image_id, results in class_results.items():
            if image_id not in all_detection_results:
                all_detection_results[image_id] = []
            all_detection_results[image_id].extend(results)
        
        map += ap
        print('ap: ', ap)
        classaps.append(ap)

    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)

    # 결과를 파일에 저장
    output_dir = 'detection_results_kari_objects_33cls'
    os.makedirs(output_dir, exist_ok=True)
    
    for image_id, results in all_detection_results.items():
        output_file = os.path.join(output_dir, f"{image_id}_det.txt")
        with open(output_file, 'w') as f:
            for result in results:
                bbox = result['bbox']
                bbox_str = ' '.join([f"{coord:.6f}" for coord in bbox])
                f.write(f"{bbox_str} {result['class_name']} {result['score']:.6f} {int(result['is_correct'])}\n")
    
    print(f"Detection results saved to {output_dir} directory")

if __name__ == '__main__':
    main()
