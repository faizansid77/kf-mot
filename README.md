# A Baseline for 3D Multi-Object Tracking

This repository contains the python implementation for "[A Baseline for 3D Multi-Object Tracking](https://arxiv.org/pdf/1907.03961.pdf)" on Waymo Open Dataset.

## Overview
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Submission](#submission)
- [Visualization](#visualization)
- [Acknowledgement](#acknowledgement)

## Introduction
3D multi-object tracking (MOT) is an essential component technology for many real-time applications such as autonomous driving or assistive robotics. However, recent works for 3D MOT tend to focus more on developing accurate systems giving less regard to computational cost and system complexity. In contrast, this work proposes a simple yet accurate real-time baseline 3D MOT system. We use an off-the-shelf 3D object detector to obtain oriented 3D bounding boxes from the LiDAR point cloud. Then, a combination of 3D Kalman filter and Mahalonobis algorithm is used for state estimation and data association.

## Dependencies
This code has been tested on python 3.7.7, and also requires the following packages:
1. scikit-learn
2. filterpy
3. matplotlib
4. pillow
5. opencv-python
6. glob2
7. tensorflow

To install required dependencies on a conda environment, please run the following command at the root of this code:
```
$ conda create -n myenv python=3.7
$ conda activate myenv
$ pip install -r requirements.txt
$ conda install tensorflow
```

## Usage
Put the dataset in "./dataset" folder.

**To convert ".tfRecord" file to ".bin" file of specific type:**

```
$ python tfRecordDataToLabel.py dataset/data.tfrecord vehicle
$ python tfRecordDataToLabel.py dataset/data.tfrecord pedestrian
$ python tfRecordDataToLabel.py dataset/data.tfrecord cyclist
```
The ".bin" files are stored in the "./dataset" folder.

**To read ".bin" data:**

```
$ python readBinFile.py dataset/data_vehicle.bin
```

**To run tracker on the ".bin" file placed in the "./dataset" folder:**

```
$ python main.py data_vehicle.bin vehicle
$ python main.py data_pedestrian.bin pedestrian
$ python main.py data_cyclist.bin cyclist
```
Then, the results will be saved to "./results" folder.

Note that, please run the code when the CPU is not occupied by other programs otherwise you might not achieve similar speed as reported in the paper.

**To preview camera stream from ".tfRecord" file:**

```
$ python showCameraData.py dataset/data.tfrecord
```

**To get the mean, std, var from dataset:**

Put all the ".tfRecord" files in "./training" folder

```
$ python getKfStats.py
```
The output is stored in dataStats.txt

## Submission
**To convert the predictions obtained to Waymo Open Dataset challenge submission format:**

Don't forget to check if "object_types" is correct in "./waymo_open_dataset/metrics/tools/submission.txtpb"

```
$ waymo_open_dataset/metrics/tools/create_submission --input_filenames='results/data_vehicle_preds.bin' --output_filename='submission/data_vehicle/model' --submission_filename='waymo_open_dataset/metrics/tools/submission.txtpb'
$ tar cvf data_vehicle.tar submission/data_vehicle
$ gzip data_vehicle.tar
```
Upload data_vehicle.tar.gz on the waymo submission server.

## Visualization
To be worked upon
 
## Acknowledgement
Part of the code is borrowed from "[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)"
