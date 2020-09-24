import sys

import tensorflow as tf
import numpy as np
from utils import load_list_from_folder
from main import iou3d, convert_3dbox_to_8corner
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, Value
from pyquaternion import Quaternion

from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.protos import metrics_pb2

OBJECT_TYPES = ['vehicle',
                'pedestrian',
                'cyclist']
LABEL_MAPPING = {label_pb2.Label.TYPE_VEHICLE: 'vehicle',
                label_pb2.Label.TYPE_PEDESTRIAN: 'pedestrian',
                label_pb2.Label.TYPE_CYCLIST: 'cyclist'}

current_object = Value('i', 0)
def init(args):
    global current_object
    current_object = args

def angle_range(rotation):
    if rotation >= np.pi: rotation -= np.pi * 2    # make the theta still in the range
    if rotation < -np.pi: rotation += np.pi * 2
    return rotation

def get_mean(gt, preds):
    global current_object

    gt_trajectory_map = {object_type: {} for object_type in OBJECT_TYPES}
    gt_box_data = {object_type: [] for object_type in OBJECT_TYPES}
    match_diff_t_map = {object_type: {} for object_type in OBJECT_TYPES}
    diff = {object_type: [] for object_type in OBJECT_TYPES} # [x, y, z, a, h, w, l]
    diff_vel = {object_type: [] for object_type in OBJECT_TYPES} # [x_dot, y_dot, z_dot, a_dot]

    for t_idx in range(len(gt.keys())):
        # print('t_idx: ', t_idx)
        t = list(gt.keys())[t_idx]
        for box_idx in range(len(gt[t])):
            # print('box_idx: ', box_idx)
            box, box_id, box_type = gt[t][box_idx]

            # [x, y, z, rz, l, w, h, 
            #  x_t - x_{t-1}, ...,  for [x,y,z,ry]
            #  (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}), ..., for [x,y,z,ry]
            box_data = np.array([
                box.center_x, box.center_y, box.center_z,
                box.heading,
                box.length, box.width, box.height, 
                0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0])


            if box_id not in gt_trajectory_map[box_type]:
                gt_trajectory_map[box_type][box_id] = {t_idx: box_data}
            else: 
                gt_trajectory_map[box_type][box_id][t_idx] = box_data

            # if we can find the same object in the previous frame, get the velocity
            if box_id in gt_trajectory_map[box_type] and t_idx-1 in gt_trajectory_map[box_type][box_id]:
                residual_vel = box_data[:4] - gt_trajectory_map[box_type][box_id][t_idx-1][:4]
                residual_vel[3] = angle_range(residual_vel[3])
                box_data[7:11] = residual_vel
                gt_trajectory_map[box_type][box_id][t_idx] = box_data
                # back fill
                if gt_trajectory_map[box_type][box_id][t_idx-1][7] == 0.0:
                    gt_trajectory_map[box_type][box_id][t_idx-1][7:11] = residual_vel

                # if we can find the same object in the previous two frames, get the acceleration
                if box_id in gt_trajectory_map[box_type] and t_idx-2 in gt_trajectory_map[box_type][box_id]:
                    residual_a = residual_vel - (gt_trajectory_map[box_type][box_id][t_idx-1][:4] - gt_trajectory_map[box_type][box_id][t_idx-2][:4])
                    residual_a[3] = angle_range(residual_a[3])
                    box_data[11:15] = residual_a
                    gt_trajectory_map[box_type][box_id][t_idx] = box_data
                    # back fill
                    if gt_trajectory_map[box_type][box_id][t_idx-1][11] == 0.0:
                        gt_trajectory_map[box_type][box_id][t_idx-1][11:15] = residual_a
                    if gt_trajectory_map[box_type][box_id][t_idx-2][11] == 0.0:
                        gt_trajectory_map[box_type][box_id][t_idx-2][11:15] = residual_a

            gt_box_data[box_type].append(box_data)
            
            with current_object.get_lock():
                current_object.value += 1
            sys.stdout.write("\r%d" % (current_object.value))
            sys.stdout.flush()

        if len(gt[t]) == 0:
            continue

        for object_type in OBJECT_TYPES:
            # print('t: ', t)
            gt_all_box = [box[0] for box in gt[t] if box[2] == object_type]
            if len(gt_all_box) == 0:
                continue
            gts = np.stack([np.array([
                box.center_x, box.center_y, box.center_z,
                box.heading,
                box.length, box.width, box.height
                ]) for box in gt_all_box], axis=0)
            gt_ids = [box[1] for box in gt[t] if box[2] == object_type]
            
            if t not in preds:
                continue
            det_all_box = [box[0] for box in preds[t] if box[1] == object_type]
            if len(det_all_box) == 0:
                continue
            dets = np.stack([np.array([
                box.center_x, box.center_y, box.center_z,
                box.heading,
                box.length, box.width, box.height
                ]) for box in det_all_box], axis=0)

            dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
            gts_8corner = [convert_3dbox_to_8corner(gt_tmp) for gt_tmp in gts]
            iou_matrix = np.zeros((len(dets_8corner),len(gts_8corner)),dtype=np.float32)
            for d,det in enumerate(dets_8corner):
                for g,gt_d in enumerate(gts_8corner):
                    iou_matrix[d,g] = iou3d(det,gt_d)
            #print('iou_matrix: ', iou_matrix)
            distance_matrix = -iou_matrix
            threshold = -0.1

            matched_indices = linear_sum_assignment(distance_matrix)
            matched_indices = np.transpose(np.asarray(matched_indices))
            #print('matched_indices: ', matched_indices)
            
            for pair_id in range(matched_indices.shape[0]):
                if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
                    diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
                    diff[object_type].append(diff_value)
                    gt_track_id = gt_ids[matched_indices[pair_id][1]]
                    if t_idx not in match_diff_t_map[object_type]:
                        match_diff_t_map[object_type][t_idx] = {gt_track_id: diff_value}
                    else:
                        match_diff_t_map[object_type][t_idx][gt_track_id] = diff_value
                    # check if we have previous time_step's matching pair for current gt object
                    #print('t: ', t)
                    #print('len(match_diff_t_map): ', len(match_diff_t_map))
                    if t_idx > 0 and t_idx-1 in match_diff_t_map[object_type] and gt_track_id in match_diff_t_map[object_type][t_idx-1]:
                        diff_vel_value = diff_value - match_diff_t_map[object_type][t_idx-1][gt_track_id]
                        diff_vel[object_type].append(diff_vel_value)

    return (gt_box_data, diff, diff_vel)

if __name__ == "__main__":
    dataset_folder = "training"
    filenames, num_files = load_list_from_folder(dataset_folder)
    context_names = [filename[filename.index("segment")+8:filename.index("with")-1] for filename in filenames]

    output_file = "dataStats.txt"

    preds_files = {'vehicle': 'dataset/detection_3d_vehicle_detection_train.bin',
                    'pedestrian': 'dataset/detection_3d_pedestrian_detection_train.bin',
                    'cyclist': 'dataset/detection_3d_cyclist_detection_train.bin'}
    
    preds = {}
    for object_type in OBJECT_TYPES:
        dataset_preds = metrics_pb2.Objects()
        with tf.io.gfile.GFile(preds_files[object_type], 'rb') as f:
            buf = f.read()
            dataset_preds.ParseFromString(buf)

        for data in dataset_preds.objects:
            if data.context_name not in context_names:
                continue
            if data.context_name not in preds:
                preds[data.context_name] = {data.frame_timestamp_micros: [(data.object.box, LABEL_MAPPING[data.object.type])]}
            else:
                if data.frame_timestamp_micros not in preds[data.context_name]:
                    preds[data.context_name][data.frame_timestamp_micros] = [(data.object.box, LABEL_MAPPING[data.object.type])]
                else:
                    preds[data.context_name][data.frame_timestamp_micros].append((data.object.box, LABEL_MAPPING[data.object.type]))

    def getContextBoxes(filename):
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        gt_dict = {}
        for data_raw in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data_raw.numpy()))
            for label in frame.laser_labels:
                if label.type not in LABEL_MAPPING.keys():
                    continue
                if frame.timestamp_micros not in gt_dict:
                    gt_dict[frame.timestamp_micros] = [(label.box, label.id, LABEL_MAPPING[label.type])]
                else:
                    gt_dict[frame.timestamp_micros].append((label.box, label.id, LABEL_MAPPING[label.type]))
        return get_mean(gt_dict, preds[frame.context.name])
    
    with Pool(initializer=init, initargs=(current_object, )) as p:
        all_stats = p.map(getContextBoxes, filenames)

    print()

    gt_stats = {object_type: np.vstack([gt_stats[0][object_type] for gt_stats in all_stats if gt_stats[0][object_type] != []]) for object_type in OBJECT_TYPES}
    diff_stats = {object_type: np.vstack([diff_stats[1][object_type] for diff_stats in all_stats if diff_stats[1][object_type] != []]) for object_type in OBJECT_TYPES}
    diff_vel_stats = {object_type: np.vstack([diff_vel_stats[2][object_type] for diff_vel_stats in all_stats if diff_vel_stats[2][object_type] != []]) for object_type in OBJECT_TYPES}

    mean_gt = {object_type: np.mean(gt_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    std_gt = {object_type: np.std(gt_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    var_gt = {object_type: np.var(gt_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    print("mean_gt:  ", mean_gt)
    print("std_gt:  ", std_gt)
    print("var_gt:  ", var_gt)

    mean_diff = {object_type: np.mean(diff_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    std_diff = {object_type: np.std(diff_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    var_diff = {object_type: np.var(diff_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    print("mean_diff:  ", mean_diff)
    print("std_diff:  ", std_diff)
    print("var_diff:  ", var_diff)

    mean_diff_vel = {object_type: np.mean(diff_vel_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    std_diff_vel = {object_type: np.std(diff_vel_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    var_diff_vel = {object_type: np.var(diff_vel_stats[object_type], axis=0) for object_type in OBJECT_TYPES}
    print("mean_diff_vel:  ", mean_diff_vel)
    print("std_diff_vel:  ", std_diff_vel)
    print("var_diff_vel:  ", var_diff_vel)

    with open(output_file, 'w') as f:
        f.write("mean_gt\n")
        f.write(str(mean_gt))
        f.write("\nstd_gt\n")
        f.write(str(std_gt))
        f.write("\nvar_gt\n")
        f.write(str(var_gt))

        f.write("\n\n\nmean_diff\n")
        f.write(str(mean_diff))
        f.write("\nstd_diff\n")
        f.write(str(std_diff))
        f.write("\nvar_diff\n")
        f.write(str(var_diff))

        f.write("\n\n\nmean_diff_vel\n")
        f.write(str(mean_diff_vel))
        f.write("\nstd_diff_vel\n")
        f.write(str(std_diff_vel))
        f.write("\nvar_diff_vel\n")
        f.write(str(var_diff_vel))