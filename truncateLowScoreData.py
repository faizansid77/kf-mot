from waymo_open_dataset.protos import metrics_pb2
import sys

def getData(filename, score):
    with open(filename, 'rb') as f:
        buf = f.read()
        labels_decoded = metrics_pb2.Objects()
        labels_decoded.ParseFromString(buf)
    out = metrics_pb2.Objects()

    for label in labels_decoded.objects:
        if label.score >= float(score):
            out.objects.append(label)

    with open(filename[:-4]+"_truncated.bin", 'wb') as f:
        f.write(out.SerializeToString())


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: python truncateLowScoreData.py dataset/data.bin 0.1")
        sys.exit(1)

    getData(sys.argv[1], sys.argv[2])