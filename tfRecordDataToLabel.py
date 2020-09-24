import tensorflow as tf
import sys

from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2, label_pb2

def tfRecordToBin(data_file_name, data_type):

    if data_type == 'vehicle':
        data_type_label = label_pb2.Label.TYPE_VEHICLE
    elif data_type == 'pedestrian':
        data_type_label = label_pb2.Label.TYPE_PEDESTRIAN
    elif data_type == 'cyclist':
        data_type_label = label_pb2.Label.TYPE_CYCLIST
    else:
        print("Usage: python tfRecordDataToLabel.py data.tfrecord vehicle")
        sys.exit(1)

    result_file_name = data_file_name[:-9] + "_" + data_type + '.bin'

    objs = metrics_pb2.Objects()

    dataset = tf.data.TFRecordDataset(data_file_name, compression_type='')
    for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for frame_obj in frame.laser_labels:
            if not frame_obj.type == data_type_label:
                continue
            obj = metrics_pb2.Object()
            obj.object.box.CopyFrom(frame_obj.box)
            obj.object.type = frame_obj.type
            obj.context_name = frame.context.name
            obj.frame_timestamp_micros = frame.timestamp_micros
            objs.objects.append(obj)

    with open(result_file_name, 'wb') as f:
        f.write(objs.SerializeToString())

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: python tfRecordDataToLabel.py data.tfrecord vehicle")
        sys.exit(1)
    
    tfRecordToBin(sys.argv[1], sys.argv[2])