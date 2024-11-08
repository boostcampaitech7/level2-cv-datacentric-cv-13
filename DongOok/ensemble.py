import json
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import os
from datetime import datetime
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='여러 모델의 결과를 앙상블하는 스크립트')
    parser.add_argument('--input_dir', required=True, 
                      help='CSV 파일들이 있는 디렉토리 경로')
    parser.add_argument('--output_dir', default='ensemble_experiments',
                      help='실험 결과를 저장할 디렉토리 경로')
    
    parser.add_argument('--iou_min', type=float, default=0.6,
                      help='실험할 최소 IOU 임계값 (기본값: 0.4)')
    parser.add_argument('--iou_max', type=float, default=0.6,
                      help='실험할 최대 IOU 임계값 (기본값: 0.6)')
    parser.add_argument('--iou_step', type=float, default=0.1,
                      help='IOU 임계값 증가 단위 (기본값: 0.1)')
    
    parser.add_argument('--vote_min', type=int, default=1,
                      help='실험할 최소 투표 수 (기본값: 1)')
    parser.add_argument('--vote_max', type=int, default=1,
                      help='실험할 최대 투표 수 (기본값: 2)')
    
    parser.add_argument('--single_iou', type=float,
                      help='단일 IOU 값으로 실험할 경우 사용')
    parser.add_argument('--single_vote', type=int,
                      help='단일 투표 수로 실험할 경우 사용')
    
    return parser.parse_args()

def calculate_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
        
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
            
        return intersection / union
    except:
        return 0.0

def ensemble_detections(csv_paths, iou_threshold=0.5, min_vote_count=1):
    all_results = []
    for path in tqdm(csv_paths, desc="파일 로딩 중"):
        with open(path, 'r') as f:
            all_results.append(json.load(f))
    
    ensemble_result = {'images': {}}
    
    all_image_names = set()
    for result in all_results:
        all_image_names.update(result['images'].keys())
    
    for image_name in tqdm(all_image_names, desc="이미지 처리 중"):
        all_boxes = []
        for model_idx, result in enumerate(all_results):
            if image_name in result['images']:
                words_info = result['images'][image_name]['words']
                for word_info in words_info.values():
                    box = word_info['points']
                    all_boxes.append({
                        'points': box,
                        'model_idx': model_idx
                    })
        
        if not all_boxes:
            continue
            
        keep_boxes = []
        while len(all_boxes) > 0:
            current_box = all_boxes.pop(0)
            overlapping_boxes = [current_box]
            
            i = 0
            while i < len(all_boxes):
                iou = calculate_iou(current_box['points'], all_boxes[i]['points'])
                if iou > iou_threshold:
                    overlapping_boxes.append(all_boxes.pop(i))
                else:
                    i += 1
            
            unique_models = len(set(box['model_idx'] for box in overlapping_boxes))
            if unique_models >= min_vote_count:
                merged_points = np.mean([box['points'] for box in overlapping_boxes], axis=0)
                keep_boxes.append(merged_points.tolist())
        
        words_info = {str(idx): {'points': box} for idx, box in enumerate(keep_boxes)}
        ensemble_result['images'][image_name] = {'words': words_info}
    
    return ensemble_result

def run_experiments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    if args.single_iou is not None and args.single_vote is not None:
        iou_thresholds = [args.single_iou]
        min_vote_counts = [args.single_vote]
    else:
        iou_thresholds = np.arange(args.iou_min, args.iou_max + args.iou_step, args.iou_step)
        min_vote_counts = range(args.vote_min, args.vote_max + 1)

    csv_paths = sorted(glob(os.path.join(args.input_dir, '*.csv')))
    if not csv_paths:
        raise ValueError(f"입력 디렉토리에서 CSV 파일을 찾을 수 없습니다: {args.input_dir}")
    
    print(f"발견된 CSV 파일 수: {len(csv_paths)}")
    for path in csv_paths:
        print(f"- {path}")

    for iou_threshold in iou_thresholds:
        for min_vote_count in min_vote_counts:
            print(f"\n실험 실행:")
            print(f"IOU 임계값: {iou_threshold:.2f}")
            print(f"최소 투표 수: {min_vote_count}")
            
            result = ensemble_detections(
                csv_paths=csv_paths,
                iou_threshold=iou_threshold,
                min_vote_count=min_vote_count
            )
            
            output_name = f"ensemble_iou{iou_threshold:.2f}_vote{min_vote_count}.csv"
            output_path = os.path.join(experiment_dir, output_name)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"결과 저장 완료: {output_path}")

def main():
    args = parse_args()
    
    print("앙상블 시작")
    run_experiments(args)
    print("앙상블 끝")

if __name__ == '__main__':
    main()