import numpy as np
import glob
import cv2
import pydicom
import csv
import matplotlib.pyplot as plt


def initialize_csv_reader(path, header):
    csv_file = open(path, mode='a', newline='')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(header)
    return writer, csv_file


def append_to_csv(record, csv_file, writer):
    writer.writerow(record)
    csv_file.flush()


def collect_images_from(path, metadata_path, csv_writer, exporting_file):
    metadata = open(file=metadata_path, mode='r')
    header = metadata.readline()
    print('dataset columns are:', header)

    records = metadata.readlines()
    image_ids = list()
    encoded_pixels = list()
    for rec in records:
        image_ids.append(rec.split(',')[0])
        encoded_pixels.append(rec.split(',')[1])

    images = list()
    for path_name in glob.glob(pathname=path + '/*/*/*.dcm'):
        image_name = path_name[path_name.rfind("\\") + 1:path_name.rfind(".")]
        for idx, image_id in enumerate(image_ids):
            if image_id == image_name:
                this_img = pydicom.dcmread(fp=path_name)
                images.append(this_img.pixel_array)
                mask = encoded_pixels[idx]

                patient_id = this_img.PatientID
                sex = this_img.PatientSex
                age = this_img.PatientAge
                view = this_img.ViewPosition

                this_record = [image_id, mask, patient_id, age, sex, view]
                append_to_csv(record=this_record, csv_file=exporting_file, writer=csv_writer)
    return np.array(images)


def save_dataset(data):
    print('number of total images:', len(data))
    print('dataset shape:', data.shape)
    print('export dataset as npy file? (y/n) \n')
    if input() == 'y':
        np.save('./siim/training_images.npy', arr=data)
    else:
        print('saving dataset ABORTED')


def main():
    csv_header = ['ImageId', 'EncodedPixels', 'PatientId', 'PatientAge', 'PatientSex', 'ImageView']
    csv_path = './siim/converted-train-rle.csv'
    writer, csv_file = initialize_csv_reader(path=csv_path, header=csv_header)

    training_path = './siim/dicom-images-train'
    md_path = './siim/train-rle.csv'
    X_train = collect_images_from(path=training_path, metadata_path=md_path, csv_writer=writer, exporting_file=csv_file)
    save_dataset(data=X_train)


if __name__ == '__main__':
    main()
