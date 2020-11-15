import numpy as np
import matplotlib.pyplot as plt
import operator
import csv


# function to extract mask from rle-coded data
def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for idx, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[idx]] = 255
        current_position += lengths[idx]

    return np.rot90(np.flip(mask.reshape(width, height), axis=1))


md_path = './siim/train-rle.csv'
image_shape = (1024, 1024)
metadata = open(file=md_path, mode='r')
records = csv.reader(metadata, delimiter=',')
sorted_records = sorted(records, key=operator.itemgetter(0))[:-1]

masks = list()
for idx, rec in enumerate(sorted_records):
    image_id = rec[0]
    encoded_pixel = rec[1]
    prev_image_id = sorted_records[idx-1][0]

    if encoded_pixel == '-1':
        # create a black mask for cases with neg label
        masks.append(np.zeros(shape=image_shape, dtype='uint8'))
    else:
        # convert RLE-coded data into mask images
        mask = rle2mask(rle=encoded_pixel, width=image_shape[0], height=image_shape[1]).astype('uint8')
        if image_id == prev_image_id:
            last_mask = masks.pop()
            masks.append(np.logical_or(mask, last_mask))
        else:
            masks.append(mask)


