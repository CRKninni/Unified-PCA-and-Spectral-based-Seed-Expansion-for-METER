import json
import sys
import os
import random
import shutil

## cmdline args handling
if len(sys.argv) < 2:
    print("Usage: test.py <sample_size>")
    sys.exit()

## load MS COCO validation qna data
f = open('valid_mega.json')
data = json.load(f)

## resuable stuff
n = int(sys.argv[1])
base_path = '../../../../data/root/val2014'
dest_path = '../../../../data/sample_val2014'

## delete existing files in the destination
existing_files = os.listdir(dest_path)
for exfile in existing_files:
    os.remove(dest_path + '/' + exfile)

## choosing n random images as sample 
images = os.listdir(base_path)
img_idxs = random.sample(range(0, len(images) - 1), n)
image_ids = []
qa_pairs = []

## copying the chosen images to a new dir
for idx in img_idxs:
    image_ids.append(images[idx][:-4])
    shutil.copy(base_path + '/' + images[idx], dest_path + '/' + images[idx])

## fetching all the corresponding qna w.r.t the chosen images
for qna in data:
    if qna['img_id'] in image_ids:
        qa_pairs.append(qna)

## dumping into json
with open('valid.json', "w") as outfile:
    json.dump(qa_pairs, outfile)


