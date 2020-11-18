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
        names = []
        for i in range(len(frame.images)):
            image = frame.images[i]

            nparr = np.frombuffer(image.image, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if i == 0:
                img_concat = img_np
            else:
                img_concat = np.concatenate((img_concat, img_np), axis=0)
            
            name = open_dataset.CameraName.Name.Name(image.name)
            names.append(name)
            
        
        cv2.namedWindow(', '.join(names), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(', '.join(names), 400, 5*300)
        cv2.imshow(', '.join(names), img_concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python showCameraData.py dataset/data.tfrecord")
        sys.exit(1)

    showCam(sys.argv[1])