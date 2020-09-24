import tensorflow as tf
import numpy as np
import sys
import cv2

from waymo_open_dataset import dataset_pb2 as open_dataset

def showCam(filename):
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        for image in frame.images:

            nparr = np.frombuffer(image.image, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            name = open_dataset.CameraName.Name.Name(image.name)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow(name, 400, 300)

            cv2.imshow(name, img_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python showCameraData.py dataset/data.tfrecord")
        sys.exit(1)

    showCam(sys.argv[1])