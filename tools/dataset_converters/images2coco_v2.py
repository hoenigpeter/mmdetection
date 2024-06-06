import argparse
import os
import json

from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, scandir, track_iter_progress
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('img_path', help='The root path of images')
    parser.add_argument(
        'classes', type=str, help='The text file name of storage class list')
    parser.add_argument(
        'out',
        type=str,
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='The suffix of images to be excluded, such as "png" and "bmp"')
    parser.add_argument(
        '-t',
        '--target-json',
        type=str,
        help='Path to the JSON file specifying target image IDs')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = scandir(path, recursive=True)
    for image_path in track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def filter_image_infos(img_infos, target_ids):
    filtered_infos = [info for idx, info in enumerate(img_infos) if idx + 1 in target_ids]
    return filtered_infos


def cvt_to_coco_json(img_infos, classes):
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes, 1):
        category_item = {
            'id': category_id,
            'name': name,
            'supercategory': 'itodd'
        }
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(os.path.splitext(os.path.basename(file_name))[0])
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

    return coco


def main():
    args = parse_args()
    assert args.out.endswith(
        'json'), 'The output file name must be json suffix'

    # Load target image IDs if specified
    target_ids = set()
    if args.target_json:
        with open(args.target_json, 'r') as f:
            targets = json.load(f)
            target_ids = {target['im_id'] for target in targets}

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, args.exclude_extensions)

    # Filter image infos based on target IDs
    if target_ids:
        img_infos = filter_image_infos(img_infos, target_ids)

    # 2 convert to coco format data
    classes = list_from_file(args.classes)
    coco_info = cvt_to_coco_json(img_infos, classes)

    # 3 dump
    save_dir = os.path.join(args.img_path, '..', 'annotations')
    mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    dump(coco_info, save_path)
    print(f'save json file: {save_path}')


if __name__ == '__main__':
    main()
