import os
from pathlib import Path
from PIL import Image
import shutil
import argparse



def main():
    print("Starting main...")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("images_path")
    # parser.add_argument("corrupted_path")

    # args = parser.parse_args()

    src_folder = str(os.getcwd())+"/NLST_dataset/NLST_jpg/"
    dst_folder = str(os.getcwd())+"/NLST_dataset/Corrupted/"

    print(os.getcwd())
    checked = 0
    n=0
    for filename in os.listdir(src_folder):
        checked+=1
        if filename.endswith('.jpg'):
            try:
                img = Image.open(src_folder+filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Error:',e,'| Bad file:', filename)
                src = src_folder + filename
                dst = dst_folder + filename
                shutil.move(src,dst)
                n+=1
        if checked % 1000==0:
            print("Checked:", checked )
            print("Found {} corrupted images".format(n))


if __name__ == '__main__':
    main()