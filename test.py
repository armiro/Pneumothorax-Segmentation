import numpy as np
import glob
import pydicom
import csv
import gzip
import cv2.cv2 as cv2


def initialize_csv_reader(path, header):
    csv_file = open(path, mode='a', newline='')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(header)
    return writer, csv_file


def append_to_csv(record, csv_file, writer):
    writer.writerow(record)
    csv_file.flush()


def collect_images_from(path, csv_writer, exporting_file):
    images = list()
    for path_name in glob.glob(pathname=path + '/*.dcm'):
        this_img = pydicom.dcmread(fp=path_name)
        images.append(cv2.resize(src=this_img.pixel_array, dsize=(512, 512)))

        image_id = path_name[path_name.rfind("\\") + 1:path_name.rfind(".")]
        patient_id = this_img.PatientID
        sex = this_img.PatientSex
        age = this_img.PatientAge
        view = this_img.ViewPosition

        this_record = [image_id, patient_id, age, sex, view]
        append_to_csv(record=this_record, csv_file=exporting_file, writer=csv_writer)

    return np.array(images)


def save_dataset(data):
    print('number of total images:', len(data))
    print('dataset shape:', data.shape)
    print('dataset size:', data.nbytes)
    print('export dataset as npy file? (y/n) \n')
    if input() == 'y':
        file = gzip.GzipFile(filename='./siim/test_images.npy.gz', mode='w')
        np.save(file, arr=data)
        file.close()
        print('compressed npy file saved!')
    else:
        print('saving dataset ABORTED!')


def main():
    csv_header = ['ImageId', 'PatientId', 'PatientAge', 'PatientSex', 'ImageView']
    csv_path = './siim/converted-test-rle.csv'
    writer, csv_file = initialize_csv_reader(path=csv_path, header=csv_header)

    test_path = './siim/dicom-images-test'
    X_test = collect_images_from(path=test_path, csv_writer=writer, exporting_file=csv_file)
    save_dataset(data=X_test)


if __name__ == '__main__':
    main()
