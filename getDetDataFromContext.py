from waymo_open_dataset.protos import metrics_pb2
import sys

def getData(filename, context_name):
    with open(filename, 'rb') as f:
        buf = f.read()
        labels_decoded = metrics_pb2.Objects()
        labels_decoded.ParseFromString(buf)
    out = metrics_pb2.Objects()
    detected = False
    for label in labels_decoded.objects:
        if label.context_name == context_name:
            detected = True
            out.objects.append(label)
        elif detected:
            break

    if not detected:
        print("Context not found!")
    else:    
        with open(filename[:-4]+"_"+context_name+".bin", 'wb') as f:
            f.write(out.SerializeToString())


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: python getDetDataFromContext.py dataset/data.bin 10203656353524179475_7625_000_7645_000")
        sys.exit(1)

    getData(sys.argv[1], sys.argv[2])