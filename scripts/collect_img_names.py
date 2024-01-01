import os
import argparse
import json

OUTPUT_FILE = os.path.join(os.path.basename(__file__), '../data/img_names_by_cat.json')

def collect_img_names_by_cat(data_dir: str, output_dir: str):
    counts = {}
    for dirpath, dirs, files in os.walk(data_dir):
        if os.path.normpath(data_dir) == dirpath: 
            continue
        counts[os.path.basename(dirpath)] = files
    with open(output_dir, 'w+') as fp:
        json.dump(counts, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-dir',
                        default=OUTPUT_FILE, 
                        type=str)
    args = parser.parse_args()
    collect_img_names_by_cat(args.data_dir, args.output_dir)