from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import njit
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils import mkdir_if_missing, fileparts
from scipy.spatial import ConvexHull
from multiprocessing import Pool, Value

from waymo_open_dataset.protos import metrics_pb2

@njit
def box3d_vol(corners):
  ''' corners: (8,3) no assumption on axis direction '''
  a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
  b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
  c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
  return a*b*c

def convex_hull_intersection(p1, p2):
  """ Compute area of two convex hull's intersection area.
      p1,p2 are a list of (x,y) tuples of hull vertices.
      return a list of (x,y) for the intersection and its volume
  """
  inter_p = polygon_clip(p1,p2)
  if inter_p is not None:
    hull_inter = ConvexHull(inter_p)
    return hull_inter.volume
  else:
    return 0.0  

def polygon_clip(subjectPolygon, clipPolygon):
  """ Clip a polygon with another polygon.
  Args:
    subjectPolygon: a list of (x,y) 2d points, any polygon.
    clipPolygon: a list of (x,y) 2d points, has to be *convex*
  Note:
    **points have to be counter-clockwise ordered**

  Return:
    a list of (x,y) vertex point for the intersection polygon.
  """
  def inside(p):
    return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
  def computeIntersection():
    dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
    dp = [ s[0] - e[0], s[1] - e[1] ]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0] 
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
  outputList = subjectPolygon
  cp1 = clipPolygon[-1]
 
  for clipVertex in clipPolygon:
    cp2 = clipVertex
    inputList = outputList
    outputList = []
    s = inputList[-1]

    for subjectVertex in inputList:
      e = subjectVertex
      if inside(e):
        if not inside(s):
          outputList.append(computeIntersection())
        outputList.append(e)
      elif inside(s):
        outputList.append(computeIntersection())
      s = e
    cp1 = cp2
    if len(outputList) == 0:
      return None
  return(outputList)

def iou3d(corners1, corners2):
  ''' Compute 3D bounding box IoU.

  Input:
      corners1: numpy array (8,3), assume up direction is negative Y
      corners2: numpy array (8,3), assume up direction is negative Y
  Output:
      iou: 3D bounding box IoU
      iou_2d: bird's eye view 2D bounding box IoU

  '''
  # corner points are in counter clockwise order
  rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3, -1, -1)]
  rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3, -1, -1)] 
  inter_area = convex_hull_intersection(rect1, rect2)
  zmin = min(corners1[0,2], corners2[0,2])
  zmax = max(corners1[4,2], corners2[4,2])
  inter_vol = inter_area * max(0.0, zmax-zmin)
  vol1 = box3d_vol(corners1)
  vol2 = box3d_vol(corners2)
  iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou

@njit
def rotz(t):
  ''' Rotation about the z-axis. '''
  c = np.cos(t)
  s = np.sin(t)
  return np.array([[c,  -s,  0.0],
                    [s,  c,  0.0],
                    [0.0, 0.0,  1.0]])

def convert_3dbox_to_8corner(bbox3d_input):
  ''' Takes an object and a projection matrix (P) and projects the 3d
      bounding box into the image plane.
      Returns:
          corners_2d: (8,2) array in left image coord.
          corners_3d: (8,3) array in in rect camera coord.
  '''
  # compute rotational matrix around yaw axis
  bbox3d = copy.copy(bbox3d_input)

  R = rotz(bbox3d[3])    

  # 3d bounding box dimensions
  l = bbox3d[4]
  w = bbox3d[5]
  h = bbox3d[6]
  
  # 3d bounding box corners
  x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
  y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
  z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]
  
  # rotate and translate 3d bounding box
  corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
  corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
  corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
  corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]

  return np.transpose(corners_3d)

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info, data_type):
    """
    Initialises a tracker using initial bounding box.
    """

    self.kf = KalmanFilter(dim_x=11, dim_z=7)       
    self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                          [0,1,0,0,0,0,0,0,1,0,0],
                          [0,0,1,0,0,0,0,0,0,1,0],
                          [0,0,0,1,0,0,0,0,0,0,1],  
                          [0,0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,1]])     
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                          [0,1,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0,0]])

    # self.kf = KalmanFilter(dim_x=10, dim_z=7)       
    # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
    #                       [0,1,0,0,0,0,0,0,1,0],
    #                       [0,0,1,0,0,0,0,0,0,1],
    #                       [0,0,0,1,0,0,0,0,0,0],  
    #                       [0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0],
    #                       [0,0,0,0,0,0,0,1,0,0],
    #                       [0,0,0,0,0,0,0,0,1,0],
    #                       [0,0,0,0,0,0,0,0,0,1]])     
    
    # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
    #                       [0,1,0,0,0,0,0,0,0,0],
    #                       [0,0,1,0,0,0,0,0,0,0],
    #                       [0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0]])

    if data_type == 'vehicle':
      #vehicle
      self.kf.Q = np.diag([3.15290085e-03,
         1.01350888e-03, 4.75131624e-03, 2.90469324e-06, 0.0, 0.0, 0.0, 3.15290085e-03,
         1.01350888e-03, 4.75131624e-03, 2.90469324e-06])
      self.kf.P = np.diag([0.11971137, 0.09417387, 0.02404045, 2.43651271, 0.25604405,
         0.02600652, 0.03556848, 0.07822007, 0.06319658, 0.00759432, 1.9484472])
      self.kf.R = np.diag([0.11971137, 0.09417387, 0.02404045, 2.43651271, 0.25604405,
         0.02600652, 0.03556848])

    elif data_type == 'pedestrian':
      #pedestrian
      self.kf.Q = np.diag([9.16921560e-04,
         4.84196197e-04, 4.15204547e-03, 2.72307475e-04, 0.0, 0.0, 0.0, 9.16921560e-04,
         4.84196197e-04, 4.15204547e-03, 2.72307475e-04])
      self.kf.P = np.diag([0.00838944, 0.00870579, 0.01922154, 5.5174705 , 0.02526426,
         0.01653421, 0.02407964, 6.85157976e-03, 7.06968962e-03, 8.24384657e-03, 4.59291252e+00])
      self.kf.R = np.diag([0.00838944, 0.00870579, 0.01922154, 5.5174705 , 0.02526426,
         0.01653421, 0.02407964])

    elif data_type == 'cyclist':
      # cyclist
      self.kf.Q = np.diag([1.70201643e-03,
        7.22589284e-04, 4.06424852e-03, 5.83792504e-05, 0.0, 0.0, 0.0, 1.70201643e-03,
        7.22589284e-04, 4.06424852e-03, 5.83792504e-05])
      self.kf.P = np.diag([0.01619721, 0.01153099, 0.02136722, 4.0544515 , 0.04284403,
        0.00994651, 0.01742469, 1.32653291e-02, 8.39735414e-03, 1.40687397e-02, 3.95335552e+00])
      self.kf.R = np.diag([0.01619721, 0.01153099, 0.02136722, 4.0544515 , 0.04284403,
        0.00994651, 0.01742469])

    else:
      print("Usage: python main.py data.bin vehicle")
      sys.exit(1)

    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info

  def update(self, bbox3D, info): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))

def angle_in_range(angle):
  '''
  Input angle: -2pi ~ 2pi
  Output angle: -pi ~ pi
  '''
  if angle > np.pi:
    angle -= 2 * np.pi
  if angle < -np.pi:
    angle += 2 * np.pi
  return angle

def diff_orientation_correction(det, trk):
  '''
  return the angle diff = det - trk
  if angle diff > 90 or < -90, rotate trk and update the angle diff
  '''
  diff = det - trk
  diff = angle_in_range(diff)
  if diff > np.pi / 2:
    diff -= np.pi
  if diff < -np.pi / 2:
    diff += np.pi
  diff = angle_in_range(diff)
  return diff

def greedy_match(distance_matrix, mahalanobis_threshold=14):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  # tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if detection_id_matches_to_tracking_id[detection_id] == -1 and distance_matrix[detection_id,tracking_id]<=mahalanobis_threshold:
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices
 
def associate_detections_to_trackers(detections,trackers, dets=None, trks=None, trks_S=None, mahalanobis_threshold=14):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)
  distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32) 
  
  assert(dets is not None)
  assert(trks is not None)
  assert(trks_S is not None)

  for d in range(len(detections)):
    for t in range(len(trackers)):
      S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
      diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
      # manual reversed angle by 180 when diff > 90 or < -90 degree
      corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
      diff[3] = corrected_angle_diff
      distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])             # det: 8 x 3, trk: 8 x 3
  
  matched_indices = greedy_match(distance_matrix)

  unmatched_detections = []
  for d in range(len(detections)):
    if len(matched_indices) == 0 or (d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t in range(len(trackers)):
    if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):
  def __init__(self, data_type, max_age=3, min_hits=2):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection   
    """              
    """
    self.data_type = data_type
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets_all):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']
    self.frame_count += 1

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    ret_info = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
    for t in reversed(to_del):
      self.trackers.pop(t)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
    trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

    if len(trks_8corner) > 0: 
      trks_8corner = np.stack(trks_8corner, axis=0)
      trks_S = np.stack(trks_S, axis=0)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, dets=dets, trks=trks, trks_S=trks_S)
    
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0], np.hstack(info[d]))

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        trk = KalmanBoxTracker(dets[i,:], info[i,:], self.data_type)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location

        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          ret_info.append(trk.info)
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret), ret_info      # x, y, z, theta, l, w, h, ID, info
    return np.empty((0,8)), np.empty((0,1))

def progress(percent, frames, total_time):
  x = int(percent * 40 // 100)
  fps = frames/total_time
  sys.stdout.write("\r[" + "#" * x + "-" * (40 - x) + "]  " + 
                    "%d%%  Frames: %d  Time: %d   FPS: %.2f" % (percent, frames, int(total_time), fps))
  sys.stdout.flush()

def endprogress(frames, total_time):
  fps = frames/total_time
  sys.stdout.write("\r[" + "#" * 40 + "]  " + 
                    "100%%  Frames: %d  Time: %d   FPS: %.2f" % (frames, int(total_time), fps))
  sys.stdout.flush()
  print()

if __name__ == '__main__':
  if len(sys.argv)!=2:
    print("Usage: python main.py data.bin vehicle")
    sys.exit(1)

  data_type = sys.argv[2]

  if data_type != 'vehicle' and data_type != 'pedestrian' and data_type != 'cyclist':
    print("Usage: python main.py data.bin vehicle")
    sys.exit(1)

  data_root = "./dataset"
  data_file_name = fileparts(sys.argv[1])[1] + fileparts(sys.argv[1])[2]

  result_root = './results'
  result_file_name = data_file_name[:-4] + "_preds.bin"

  data_file = os.path.join(data_root, data_file_name)
  mkdir_if_missing(result_root)

  dataset = metrics_pb2.Objects()
  
  with open(data_file, 'rb') as f:
    buf = f.read()
    dataset.ParseFromString(buf)

  contexts = []
  if len(dataset.objects) != 0:
    context = dataset.objects[0].context_name
    start = 0
    i = 1
    for data in dataset.objects[1:]:
      if data.context_name != context:
        contexts.append((start, i))
        context = data.context_name
        start = i
      i += 1
    contexts.append((start, i))

  total_objects = len(dataset.objects)
  current_object = Value('i', 0)
  start_time = time.time()
  
  def init(args):
    global current_object
    current_object = args

  def solve(context):
    global current_object
    (start, end) = context
    mot_tracker = AB3DMOT(data_type=data_type)
    ids = []

    dets = []
    info = []
    frame = dataset.objects[0].frame_timestamp_micros

    obj_no = 0
    for data in dataset.objects[start:end]:
      if data.frame_timestamp_micros != frame:
        all_dets = {'dets': np.array(dets), 'info': np.array(info)}
        trackers, info_t = mot_tracker.update(all_dets)
        for d in range(len(trackers)):
          for md in info_t[d]:
            ids.append([int(md), str(int(trackers[d][7]))])

        dets = []
        info = []
        frame = data.frame_timestamp_micros

      box = data.object.box
      dets.append([box.center_x, box.center_y, box.center_z, box.heading, box.length, box.width, box.height])
      info.append([start+obj_no])
      with current_object.get_lock():
        current_object.value += 1
      progress(100 * current_object.value//total_objects, current_object.value, time.time() - start_time)

      obj_no += 1
      
    all_dets = {'dets': np.array(dets), 'info': np.array(info)}
    trackers, info_t = mot_tracker.update(all_dets)
    for d in range(len(trackers)):
      for md in info_t[d]:
        ids.append([int(md), str(int(trackers[d][7]))])
    return ids

  with Pool(initializer=init, initargs=(current_object, )) as p:
    all_ids = p.map(solve, contexts)
  
  total_time = time.time() - start_time

  for context_ids in all_ids:
    for frame_id in context_ids:
      dataset.objects[frame_id[0]].object.id = frame_id[1]

  endprogress(current_object.value, total_time)

  with open(os.path.join(result_root, result_file_name), 'wb') as f:
    f.write(dataset.SerializeToString())