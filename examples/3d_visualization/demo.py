#!/usr/bin/env python3

import sys
sys.path.append("c:\\Users\\axlab\\OneDrive\\Desktop\\Projects\\SHAPE-IT\\depthai_hand_tracker")
from HandTrackerRenderer import HandTrackerRenderer
from Filters import LandmarksSmoothingFilter, OneEuroFilter
import argparse
import numpy as np
import cv2
from o3d_utils import Visu3D
class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.threshold = 0.1
        self.prev_value = None

    def __call__(self, value):
        if self.prev_value is None:
            self.prev_value = value
        else:
            if abs(value - self.prev_value) > self.threshold:
                return self.prev_value
                
            else:
                self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        return self.prev_value

import socket
from itertools import chain

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTracker3DRenderer:
    def __init__(self, tracker, mode_3d="image", smoothing=True):

        self.tracker = tracker
        self.mode_3d = mode_3d
        if self.mode_3d == "mixed" and not self.tracker.xyz:
            print("'mixed' 3d visualization needs the tracker to be in 'xyz' mode !")
            print("3d visualization falling back to 'world' mode.")
            self.mode_3d = 'world'
        if self.mode_3d == "image":
            self.vis3d = Visu3D(zoom=0.7, segment_radius=10)
            z = min(tracker.img_h, tracker.img_w)/3
            self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
            self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
            self.vis3d.init_view()
        elif "world" in self.mode_3d:
            self.vis3d = Visu3D(bg_color=(0.2, 0.2, 0.2), zoom=1.1, segment_radius=0.01)
            x_max = 0.2 if self.tracker.solo else 0.4
            y_max = 0.2
            z_max = 0.2
            self.vis3d.create_grid([-x_max,y_max,-z_max],[x_max,y_max,-z_max],[x_max,y_max,z_max],[-x_max,y_max,z_max],1 if self.tracker.solo else 2,1) # Floor
            self.vis3d.create_grid([-x_max,y_max,z_max],[x_max,y_max,z_max],[x_max,-y_max,z_max],[-x_max,-y_max,z_max],1 if self.tracker.solo else 2,1) # Wall
            self.vis3d.init_view()
        elif self.mode_3d == "mixed":
            self.vis3d = Visu3D(bg_color=(0.4, 0.4, 0.4), zoom=0.8, segment_radius=0.01)
            x_max = 0.9
            y_max = 0.6
            grid_depth = 2
            self.vis3d.create_grid([-x_max,y_max,0],[x_max,y_max,0],[x_max,y_max,grid_depth],[-x_max,y_max,grid_depth],2,grid_depth) # Floor
            self.vis3d.create_grid([-x_max,y_max,grid_depth],[x_max,y_max,grid_depth],[x_max,-y_max,grid_depth],[-x_max,-y_max,grid_depth],2,2) # Wall
            self.vis3d.create_camera()
            self.vis3d.init_view()

        self.smoothing = smoothing
        self.filter = None
        if self.smoothing:
            if tracker.solo:
                if self.mode_3d == "image":
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=0.01, beta=40, derivate_cutoff=1)]
                else:
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)]
            else:
                if self.mode_3d == "image":
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1),
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1)
                        ]
                else:
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True),
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)
                        ]

        self.nb_hands_in_previous_frame = 0

    def draw_hand(self, hand, i):
        if self.mode_3d == "image":
            # Denormalize z-component of 'norm_landmarks'
            lm_z = (hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4).astype(np.int32)
            # ... and concatenates with x and y components of 'landmarks'
            points = np.hstack((hand.landmarks, lm_z))
            radius = hand.rect_w_a / 30 # Thickness of segments depends on the hand size
        elif "world" in self.mode_3d:
            if self.mode_3d == "raw_world":
                points = hand.world_landmarks
            else: # "world"
                points = hand.get_rotated_world_landmarks()
            if not self.tracker.solo:
                delta_x = -0.2 if  hand.label == "right" else 0.2
                points = points + np.array([delta_x,0,0])
            radius = 0.01
        elif self.mode_3d == "mixed":
            wrist_xyz = hand.xyz / 1000.0
            # Beware that y value of (x,y,z) coordinates given by depth sensor is negative 
            # in the lower part of the image and positive in the upper part.
            wrist_xyz[1] = -wrist_xyz[1]
            points = hand.get_rotated_world_landmarks()
            points = points + wrist_xyz - points[0]
            radius = 0.01

        if self.smoothing:
            points = self.filter[i].apply(points, object_scale=hand.rect_w_a)

        for i,a_b in enumerate(LINES_HAND):
            a, b = a_b
            self.vis3d.add_segment(points[a], points[b], radius=radius, color=[1*(1-hand.handedness),hand.handedness,0]) # if hand.handedness<0.5 else [0,1,0])
                    
    def draw(self, hands):
        if self.smoothing and len(hands) != self.nb_hands_in_previous_frame:
            for f in self.filter: f.reset()
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        for i, hand in enumerate(hands):
            self.draw_hand(hand, i)
        self.vis3d.render()
        self.nb_hands_in_previous_frame = len(hands)

parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--edge', action="store_true",
#                     help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Path to a blob file for landmark model")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Detect one hand max")         
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")    
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=0,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")                

parser_renderer3d = parser.add_argument_group("3D Renderer arguments")
parser_renderer3d.add_argument('-m', '--mode_3d', nargs='?', 
                    choices=['image', 'world', 'raw_world', 'mixed'], const='image', default='image',
                    help="Specify the 3D coordinates used. See README for description (default=%(default)s)")
parser_renderer3d.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter (smoothing works only in solo mode)")   

parser_renderer3d.add_argument('--send_data', action="store_true")  
args = parser.parse_args()

args.edge = True
if args.edge:
    from HandTrackerEdge import HandTracker
else:
    from HandTracker import HandTracker

if args.send_data:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5058)


dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandTracker(
        input_src=args.input, 
        use_world_landmarks=args.mode_3d != "image",
        solo=args.solo,
        xyz= args.mode_3d == "mixed",
        stats=True,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=1,
        **tracker_args
        )

renderer3d = HandTracker3DRenderer(tracker, mode_3d=args.mode_3d, smoothing=not args.no_smoothing)
renderer2d = HandTrackerRenderer(tracker)

pause = False
hands = []

#initialize EMAs
alpha = 0.5
filters0 = [EMAFilter(alpha=alpha) for _ in range(63)]
filters1 = [EMAFilter(alpha=alpha) for _ in range(63)]

while True:
    
    # Run hand tracker on next frame
    if not pause:
        print("#################################")
        frame, hands, bag = tracker.next_frame()
        data = ""
        if frame is None: break
        # Render 2d frame
        if hands:
            data = []
            lmList = []
            lmList1 = []

            hand0=hands[0] # get first hand detected
            hand0Type = hand0.label
            lmList.append(hand0Type)

            wrist_xyz0 = hand0.xyz / 1000.0
            wrist_xyz0[1] = -wrist_xyz0[1]
            points0 = hand0.get_rotated_world_landmarks()
            points0 = points0 + wrist_xyz0

            #EMA filter
            pointList0 = list(chain.from_iterable(points0))
            for i, num in enumerate(chain.from_iterable(points0)):
                pointList0[i] = filters0[i](num)


            lmList.append(list(round(num, 5) for num in pointList0)) # Get the landmark list
            print(list(round(num, 6) for num in pointList0))

            if len(hands) == 1:
                data = lmList

            elif len(hands) == 2:
                hand1=hands[1]
                hand1Type = hand1.label
                lmList1.append(hand1Type)

                wrist_xyz1 = hand1.xyz / 1000.0
                wrist_xyz1[1] = -wrist_xyz1[1]
                points1 = hand1.get_rotated_world_landmarks()
                points1 = points1 + wrist_xyz1

                pointList1 = list(chain.from_iterable(points1))
                for i, num in enumerate(chain.from_iterable(points1)):
                    pointList1[i] = filters1[i](num)

                lmList1.append(list(round(num, 5) for num in pointList1))

                if hand1Type == "left":
                    data = lmList1
                    data.extend(lmList)
                else:
                    data = lmList
                    data.extend(lmList1)

        #send data with lower accuracy
        #print(data)
        

        if args.send_data:
            sock.sendto(str.encode(str(data)), serverAddressPort)
        frame = renderer2d.draw(frame, hands, bag)
        cv2.imshow("HandTracker", frame)
    key = cv2.waitKey(1)
    # Draw hands on open3d canvas
    #renderer3d.draw(hands)
    if key == 27 or key == ord('q'):
        break
    elif key == 32: # space
        pause = not pause
    elif key == ord('s'):
        if renderer3d.filter:
            renderer3d.smoothing = not renderer3d.smoothing
renderer2d.exit()
tracker.exit()
