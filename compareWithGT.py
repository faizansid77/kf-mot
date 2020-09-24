from waymo_open_dataset.protos import metrics_pb2
import sys

def compare(filename, filename_gt):
    with open(filename, 'rb') as f:
        buf = f.read()
        dataset = metrics_pb2.Objects()
        dataset.ParseFromString(buf)

    with open(filename_gt, 'rb') as f:
        buf = f.read()
        dataset_gt = metrics_pb2.Objects()
        dataset_gt.ParseFromString(buf)

    names = {}
    global num_ctr
    num_ctr = 0

    def assignNum(name):
        global num_ctr
        if name in names:
            return names[name]
        num_ctr += 1
        names[name] = num_ctr

    unknown = 0
    correct = 0
    incorrect = 0

    print('data', "\t", 'gt')
    for i in range(max(len(dataset.objects), len(dataset_gt.objects))):
        if i < len(dataset.objects):
            label = dataset.objects[i].object.id
        else:
            label = "done"
        if i < len(dataset_gt.objects):
            label_gt = assignNum(dataset_gt.objects[i].object.id)
        else:
            label_gt = "done"
        if label == "":
            label = None
        if label_gt == None or label == None:
            unknown += 1
        elif label == str(label_gt):
            correct += 1
        else:
            incorrect += 1
        print(label, "\t", label_gt)

    print(correct, incorrect, unknown)


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: python compareWithGT.py data.bin data_gt.bin")
        sys.exit(1)
    
    compare(sys.argv[1], sys.argv[2])