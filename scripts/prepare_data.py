import os
import shutil
import argparse

def prepare_data(imgs_path: str):
    _, directories, files = next(os.walk(imgs_path))
    created_dirs = set(directories)

    for file in files:
        file_name, ext = os.path.splitext(file)
        prefix, dset, id, cat = file_name.split('_')
        new_img_name = f'{prefix}_{dset}_{id}{ext}'

        if cat not in created_dirs:
            os.mkdir(os.path.join(imgs_path, cat))
            created_dirs.add(cat)
        
        shutil.move(os.path.join(imgs_path, file), os.path.join(imgs_path, cat, new_img_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', 
                        default=os.path.join(os.path.dirname(__file__), 'val'),
                        type=str)
    args = parser.parse_args()
    prepare_data(args.images_path)