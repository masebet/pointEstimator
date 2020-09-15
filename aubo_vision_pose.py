#import appropriate python modules to the program

#import freenect
import socket
import os.path
import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
import geometry_msgs.msg
from aubo_i5_moveit_config.msg import Tracker
from aubo_i5_moveit_config.msg import Centroid
import moveit_msgs.msg
import cv2, cv_bridge
from sensor_msgs.msg import Image
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
#import freenect


from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
tracker = Tracker()
centroid = Centroid()


# capturing video from Kinect Xbox 360
'''def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array'''

# callback function for selecting object by clicking 4-corner-points of the object
def select_object(event, x, y, flags, param):
    global box_pts, frame
    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(box_pts) < 4:
        box_pts.append([x, y])
        frame = cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)

# selecting object by clicking 4-corner-points
def select_object_mode():
    global input_mode, initialize_mode
    input_mode = True
    
    frame_static = frame.copy()

    while len(box_pts) < 4:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    
    initialize_mode = True
    input_mode = False

# setting the boundary of reference object
def set_boundary_of_reference(box_pts):
    
    ### upper bound ###
    if box_pts[0][1] < box_pts[1][1]:
        upper_bound = box_pts[0][1]
    else:
        upper_bound = box_pts[1][1]
    
    ### lower bound ###
    if box_pts[2][1] > box_pts[3][1]:
        lower_bound = box_pts[2][1]
    else:
        lower_bound = box_pts[3][1]
    
    ### left bound ###
    if box_pts[0][0] < box_pts[2][0]:
        left_bound = box_pts[0][0]
    else:
        left_bound = box_pts[2][0]
    
    ### right bound ###
    if box_pts[1][0] > box_pts[3][0]:
        right_bound = box_pts[1][0]
    else:
        right_bound = box_pts[3][0]
        
    upper_left_point = [0,0]
    upper_right_point = [(right_bound-left_bound),0]
    lower_left_point = [0,(lower_bound-upper_bound)]
    lower_right_point = [(right_bound-left_bound),(lower_bound-upper_bound)]
    
    pts2 = np.float32([upper_left_point, upper_right_point, lower_left_point, lower_right_point])
    
    # display dimension of reference object image to terminal
    print pts2
    
    return pts2, right_bound, left_bound, lower_bound, upper_bound

# doing perspective transform to reference object
def input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound):
    global object_orb
    pts1 = np.float32(box_pts)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_object = cv2.warpPerspective(frame,M,((right_bound-left_bound),(lower_bound-upper_bound)))
    return cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

# feature detection and description using ORB
def orb_feature_descriptor(img_object):
    kp1, des1 = orb.detectAndCompute(img_object,None)
    kp2, des2 = orb.detectAndCompute(frame,None)
    return kp1, des1, kp2, des2

# feature matching using Brute Force
def brute_force_feature_matcher(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key = lambda x:x.distance)
    
def sift_force_feature_matcher(kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = []
    try:
		matche_raw = flann.knnMatch(des1, des2, k=2)
		# store all the good matches as per Lowe's ratio test.
    except:
		return matches
    for m, n in matche_raw:
		if m.distance < 0.75 * n.distance:
			matches.append(m)
    return matches

# finding homography matrix between reference and image frame
def find_homography_object(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M, mask

# applying homography matrix as inference of perpective transformation
def output_perspective_transform(img_object, M):
    h,w = img_object.shape
    corner_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
    corner_pts_3d = np.float32([ [-w/2,-h/2,0],[-w/2,(h-1)/2,0],[(w-1)/2,(h-1)/2,0],[(w-1)/2,-h/2,0] ])
    corner_camera_coord = cv2.perspectiveTransform(corner_pts,M)
    center_camera_coord = cv2.perspectiveTransform(center_pts,M)
    return corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts

# solving pnp using iterative LMA algorithm
def iterative_solve_pnp(object_points, image_points, kinect_intrinsic_param, kinect_distortion_param):
    image_points = image_points.reshape(-1,2)
    retval, rotation, translation = cv2.solvePnP(object_points, image_points, kinect_intrinsic_param, kinect_distortion_param)
    return rotation, translation

# drawing box around object
def draw_box_around_object(dst):
    return cv2.polylines(frame, [np.int32(dst)],True,255,3)
    
# recording sample data
def record_samples_data(translation, rotation):
    translation_list = translation.tolist()
    rotation_list = rotation.tolist()
    
    t1.append(translation_list[0])
    t2.append(translation_list[1])
    t3.append(translation_list[2])
    
    r1.append(rotation_list[0])
    r2.append(rotation_list[1])
    r3.append(rotation_list[2])
    
# computing and showing recorded data to terminal
def showing_recorded_data_to_terminal(t1, t2, t3, r1, r2, r3):
    
    # convert to numpy array
    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)
    
    # print mean and std of the data to terminal
    print "mean t1", np.mean(t1)
    print "std t1", np.std(t1)
    print ""
    print "mean t2", np.mean(t2)
    print "std t2", np.std(t2)
    print ""
    print "mean t3", np.mean(t3)
    print "std t3", np.std(t3)
    print ""
    print ""
    print "mean r1", np.mean(r1)
    print "std r1", np.std(r1)
    print ""
    print "mean r2", np.mean(r2)
    print "std r2", np.std(r2)
    print ""
    print "mean r3", np.mean(r3)
    print "std r3", np.std(r3)
    print ""
    print "#####################"
    print ""

# showing object position and orientation value to frame
def put_position_orientation_value_to_frame(translation, rotation):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,'position(M)',(10,30), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(translation[0],4)),(250,30), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(translation[1],4)),(350,30), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(translation[2],4)),(450,30), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    
    cv2.putText(frame,'orientation(degree)',(10,60), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(rotation[0],2)),(250,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(rotation[1],2)),(350,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(rotation[2],2)),(450,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)

    cv2.putText(frame, 'Find matches: ',(10,90), font, 0.7, (0, 255, 0),1,cv2.LINE_AA)

    cv2.putText(frame, "%s/%s" % (len(matches), 40), (250, 90), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    
    
    return frame


############
### Main ###
############

# initialization
input_mode = False
initialize_mode = False
track_mode = False
box_pts = []

nilai = []

#master
'''lis = socket.socket()
#deklarasi serverIp dan serverport
serverIP = "127.0.0.1"
serverPort = 8888
lis.connect((serverIP,serverPort))'''

record_num = 0
record_mode = False

t1, t2, t3, r1, r2, r3 = [], [], [], [], [], []

#kinect_intrinsic_param = np.array([[514.04093664, 0., 320], [0., 514.87476583, 240], [0., 0., 1.]])
#kinect_distortion_param = np.array([2.68661165e-01, -1.31720458e+00, -3.22098653e-03, -1.11578383e-03, 2.44470018e+00])
#kinect_intrinsic_param = np.array([[523.0881374691555, 0, 315.548196440562], [0, 523.6917019605472, 262.7348496146574], [0, 0, 1]])
#kinect_distortion_param = np.array([0.1385122897132973, -0.2616246386412429, 0.001305311377419195, 0.0005540614144078647, 0])

#kinnect
#kinect_intrinsic_param = np.array([[592.279193124937, 0, 311.99525405847], [0, 590.8417039104305, 200.2525788128476], [0, 0, 1]])
#kinect_distortion_param = np.array([-0.1063090312654929, 0.2314799809179102, -0.03745035406455523, -0.008998086620084576, 0])

#webcam
#kinect_intrinsic_param = np.array([[730.6769382867512, 0, 351.0711604199586], [0, 726.8491564397995, 252.9109424008911], [0, 0, 1]])
#kinect_distortion_param = np.array([0.05326230789799909, -0.4935000032080154, -0.001064348623290248, -0.002935789163374075, 0])

#webcam 4K lenovo
#kinect_intrinsic_param = np.array([[497.221667, 0.000000, 320.939509], [0.000000, 496.950484, 255.048683], [0, 0, 1]])
#kinect_distortion_param = np.array([0.169255, -0.320682, 0.002905, -0.001279, 0])

#webcam 4K dell
kinect_intrinsic_param = np.array([[485.4997921509626, 0, 322.4100183573505], [0, 483.5401052141448, 253.2432060002476], [0, 0, 1]])
kinect_distortion_param = np.array([0.07639549532234714, -0.2339693845685686, 0.007033113364483832, -0.002724655720445485, 0])

orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()
bridge = cv_bridge.CvBridge()
#cap=cv2.VideoCapture(1)
cap = WebcamVideoStream(src=1).start()
fps = FPS().start()
#cap.set(3,320)
#cap.set(4,240)


cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_object)

'''rospy.init_node("ur5_vision", anonymous=False)
track_flag = False
default_pose_flag = True
cx = 400.0
cy = 400.0
bridge = cv_bridge.CvBridge()
#image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
cxy_pub = rospy.Publisher('cxy', Tracker, queue_size=1)'''

while 1:
        rospy.init_node("ur5_vision", anonymous=False)
        cxy_pub = rospy.Publisher('cxy', Tracker, queue_size=1)
        #rate = rospy.Rate(10)
	frame = cap.read()                              #webcam
	#frame = get_video()                             #kinnect
	#frame = imutils.resize(frame, width=480)
    	#cv2.line(frame, (0,240), (640,240), (255,0,0), 1)
    	#cv2.line(frame, (320,0), (320,480), (0,0,250), 1)
    	#track_mode = False

	k = cv2.waitKey(30) & 0xFF

	

	if k == ord('l'):

	    img_object = cv2.imread('pixy.jpg',0)
	    img_object2 = cv2.imread('wifi.jpg',0)
	    img_object3 = cv2.imread('roko.jpg',0)
	    print 'load'
		    
	    track_mode = True
	    kp1, des1 = sift.detectAndCompute(img_object, None)
	    kp3, des3 = sift.detectAndCompute(img_object2, None)
	    kp5, des5 = sift.detectAndCompute(img_object3, None)
	    #cv2.imshow('load', img_object)

	if k == ord('p'):

	    img_object = cv2.imread('roko.jpg',0)
	    print 'load'
		    
	    track_mode = True
	    kp1, des1 = sift.detectAndCompute(img_object, None)
	    #cv2.imshow('load', img_object)


	# press i to enter input mode
	if k == ord('i'):
	    print 'det'
	    # select object by clicking 4-corner-points
	    select_object_mode()
	    
	    
	    # set the boundary of reference object
	    pts2, right_bound, left_bound, lower_bound, upper_bound = set_boundary_of_reference(box_pts)
	    
	    # do perspective transform to reference object
	    img_object = input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound)
	    #cv2.imshow('load', img_object)
	    
	    track_mode = True
	    kp1, des1 = sift.detectAndCompute(img_object, None)

	if k == ord('s'):
		print"save image"
		cv2.imwrite('pixy_bot.jpg', img_object)

	# track mode is run immediately after user selects 4-corner-points of object
	if track_mode is True:
	    # feature detection and description
	    #kp1, des1, kp2, des2 = orb_feature_descriptor(img_object)
	    kp2, des2 = sift.detectAndCompute(frame, None)
	    kp4, des4 = sift.detectAndCompute(frame, None)
	    kp6, des6 = sift.detectAndCompute(frame, None)

	    if k == ord('o'):
                track_mode = False
                print 'unload'
	    
	    # feature matching
	    #matches = brute_force_feature_matcher(kp1, des1, kp2, des2)
	    matches = sift_force_feature_matcher(kp1, des1, kp2, des2)
	    matches2 = sift_force_feature_matcher(kp3, des3, kp4, des4)
	    matches3 = sift_force_feature_matcher(kp5, des5, kp6, des6)
	    #print len(matches)
	    if len(matches) > 40:
	    # find homography matrix
		track_flag = True
		M, mask = find_homography_object(kp1, kp2, matches)
		
		# apply homography matrix using perspective transformation
		corner_camera_coord, center_camera_coord, object_points_3d, center_pts = output_perspective_transform(img_object, M)
		
		# solve pnp using iterative LMA algorithm
		rotation, translation = iterative_solve_pnp(object_points_3d, corner_camera_coord, kinect_intrinsic_param, kinect_distortion_param)

		#print"translasi =",translation
		#print"rotasi =",rotation
		
		# convert to centimeters
		translation = (((40./53.) * translation *.1)/3) #0.0754716981132*translation
		
		
		#lis.send('\n')
		
		# convert to degree
		rotation = rotation * 180./np.pi
		
		#translation_list = translation.tolist()
		#rotation_list = rotation.tolist()

		tX = str(round(translation[0],4))
		tY = str(round(translation[1],4))
		tZ = str(round(translation[2],4))

		rX = str(round(rotation[0],4))
		rY = str(round(rotation[1],4))
		rZ = str(round(rotation[2],4))

		#centroid.error_x = float (tX)
		#centroid.error_y = float (tY)

		tracker.flag1 = track_flag
		#tracker.x = tX
		#tracker.y = tY
		#tracker.z = tZ
		tracker.error_x = float (tX)
		tracker.error_y = float (tY)
		tracker.error_z = float (tZ)
		

		'''print ('error_x =', tX)
		print ('error_y =', tY)
		print ('error_z =', tZ)'''
		                    
		
		nilai = tX, tY, tZ, rX, rY, rZ
		#print len(nilai)

		
		#n = str(nilai)
		#lis.send(n)
		                                                                       
		# draw box around object
		frame = draw_box_around_object(corner_camera_coord)
		
		# show object position and orientation value to frame
		frame = put_position_orientation_value_to_frame(translation, rotation)

            
	    elif len(matches2) > 40:
	    # find homography matrix
		track_flag = True
		M2, mask = find_homography_object(kp3, kp4, matches2)
		
		# apply homography matrix using perspective transformation
		corner_camera_coord, center_camera_coord, object_points_3d, center_pts = output_perspective_transform(img_object2, M2)
		
		# solve pnp using iterative LMA algorithm
		rotation, translation = iterative_solve_pnp(object_points_3d, corner_camera_coord, kinect_intrinsic_param, kinect_distortion_param)

		#print"translasi =",translation
		#print"rotasi =",rotation
		
		# convert to centimeters
		#translation = (((40./53.) * translation *.1)/3) #0.0754716981132*translation
		
		
		#lis.send('\n')
		
		# convert to degree
		rotation = rotation * 180./np.pi
		
		#translation_list = translation.tolist()
		#rotation_list = rotation.tolist()

		tX = str(round(translation[0],4))
		tY = str(round(translation[1],4))
		tZ = str(round(translation[2],4))

		rX = str(round(rotation[0],4))
		rY = str(round(rotation[1],4))
		rZ = str(round(rotation[2],4))

		#centroid.error_x = float (tX)
		#centroid.error_y = float (tY)

		tracker.flag1 = track_flag
		#tracker.x = tX
		#tracker.y = tY
		#tracker.z = tZ
		tracker.error_x = float (tX)
		tracker.error_y = float (tY)
		tracker.error_z = float (tZ)
		

				                    
		
		nilai = tX, tY, tZ, rX, rY, rZ
		#print len(nilai)

		
		#n = str(nilai)
		#lis.send(n)
		                                                                       
		# draw box around object
		frame = draw_box_around_object(corner_camera_coord)
		
		# show object position and orientation value to frame
		frame = put_position_orientation_value_to_frame(translation, rotation)

	    elif len(matches3) > 40:
	    # find homography matrix
		track_flag = True
		M3, mask = find_homography_object(kp5, kp6, matches3)
		
		# apply homography matrix using perspective transformation
		corner_camera_coord, center_camera_coord, object_points_3d, center_pts = output_perspective_transform(img_object3, M3)
		
		# solve pnp using iterative LMA algorithm
		rotation, translation = iterative_solve_pnp(object_points_3d, corner_camera_coord, kinect_intrinsic_param, kinect_distortion_param)

		#print"translasi =",translation
		#print"rotasi =",rotation
		
		# convert to centimeters
		#translation = (((40./53.) * translation *.1)/3) #0.0754716981132*translation
		
		
		#lis.send('\n')
		
		# convert to degree
		rotation = rotation * 180./np.pi
		
		#translation_list = translation.tolist()
		#rotation_list = rotation.tolist()

		tX = str(round(translation[0],4))
		tY = str(round(translation[1],4))
		tZ = str(round(translation[2],4))

		rX = str(round(rotation[0],4))
		rY = str(round(rotation[1],4))
		rZ = str(round(rotation[2],4))

		#centroid.error_x = float (tX)
		#centroid.error_y = float (tY)

		tracker.flag1 = track_flag
		#tracker.x = tX
		#tracker.y = tY
		#tracker.z = tZ
		tracker.error_x = float (tX)
		tracker.error_y = float (tY)
		tracker.error_z = float (tZ)
		

				                    
		
		nilai = tX, tY, tZ, rX, rY, rZ
		#print len(nilai)

		
		#n = str(nilai)
		#lis.send(n)
		                                                                       
		# draw box around object
		frame = draw_box_around_object(corner_camera_coord)
		
		# show object position and orientation value to frame
		frame = put_position_orientation_value_to_frame(translation, rotation)

	    else:
		track_flag = False
		tracker.flag1 = track_flag
	    


	cxy_pub.publish(tracker)
	#cv2.namedWindow("window", 1)
	cv2.imshow("frame", frame)
	fps.update()
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	   

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.stop()
#cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows() 
#rospy.spin()
