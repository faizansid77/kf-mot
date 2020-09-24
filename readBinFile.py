from waymo_open_dataset.protos import metrics_pb2
import sys

def readBin(filename):
    with open(filename, 'rb') as f:
        buf = f.read()
        labels_decoded = metrics_pb2.Objects()
        labels_decoded.ParseFromString(buf)
    # frame = labels_decoded.objects[0].frame_timestamp_micros
    for label in labels_decoded.objects:
        # if label.frame_timestamp_micros != frame:
        #     break
        print(label)

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Usage: python readBinFile.py dataset/data.bin")
        sys.exit(1)

    readBin(sys.argv[1])