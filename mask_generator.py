import numpy as np
import operator
import csv
import gzip
import cv2.cv2 as cv2
import matplotlib.pyplot as plt


# function to extract mask from rle-coded data
def rle2mask(rle, width=1024, height=1024):
    # because of the range of the numbers in the dataset masks column, we have to create default (1024,1024) masks
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


def generate_masks_from(metadata_path, mask_shape):
    metadata = open(file=metadata_path, mode='r')
    records = csv.reader(metadata, delimiter=',')
    sorted_records = sorted(records, key=operator.itemgetter(0))[:-1]

    masks = list()
    for idx, rec in enumerate(sorted_records):
        image_id = rec[0]
        encoded_pixel = rec[1]
        prev_image_id = sorted_records[idx - 1][0]

        if encoded_pixel == '-1':
            # create a black mask for cases with neg label
            masks.append(np.zeros(shape=mask_shape, dtype='uint8'))
        else:
            # convert RLE-coded data into mask images, and reshape them
            mask = cv2.resize(src=rle2mask(rle=encoded_pixel).astype('uint8'), dsize=mask_shape)
            if image_id == prev_image_id:
                last_mask = masks.pop()
                masks.append(np.logical_or(mask, last_mask))
            else:
                masks.append(mask)
    return np.array(masks)


def save_masks(masks):
    print('number of images in the dataset with at least one corresponding mask:', len(masks))
    print('shape of generated ground-truth:', masks.shape)
    print('size of generated ground-truth:', masks.nbytes)
    print('save the generated masks? (y/n)')
    if input() == 'y':
        file = gzip.GzipFile(filename='./siim/training_masks_resized.npy.gz', mode='w')
        np.save(file, arr=masks)
        file.close()
        print('compressed npy file saved!')
    else:
        print('saving masks ABORTED!')


def main():
    md_path = './siim/train-rle.csv'
    generated_masks = generate_masks_from(metadata_path=md_path, mask_shape=(512, 512))
    save_masks(masks=generated_masks)


if __name__ == '__main__':
    main()
