import os
import shutil
import argparse

def prepare_data(imgs_path: str):
    
    [[os.makedirs(os.path.join(imgs_path, cat, iterations)) 
      for iterations in ['250', '500', '750']]
      for cat in ['clear', 'div']]

    _, _, files = next(os.walk(imgs_path))

    for file in files:
        
        file_name, ext = os.path.splitext(file)
        file_path = os.path.join(imgs_path, file)

        fn_parts = file_name.split('_')
        batch_ind = None
        if len(fn_parts) == 3:
            cls, cat, iterations = fn_parts
        else:
            cls, cat, _, batch_ind, iterations = fn_parts

        new_img_name = f'{cls}{("_" + batch_ind) if batch_ind is not None else ""}_{iterations}{ext}'
        new_img_path = os.path.join(imgs_path, cat, iterations, new_img_name)
        
        shutil.move(file_path, new_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', 
                        default=os.path.join(os.path.dirname(__file__), 'images'),
                        type=str)
    args = parser.parse_args()
    prepare_data(args.images_path)