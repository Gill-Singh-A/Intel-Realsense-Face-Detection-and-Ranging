#!/usr/bin/env python3

import rospy, cv2, numpy, math, image_geometry
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point

bridge = CvBridge()
camera_model = image_geometry.PinholeCameraModel()

cameraInfo = None
image = None
depth_image = None
ray = None

GREEN = (0, 255, 0)

scale_factor = 1.3
min_neighbors = 5
cascade_file = "xml/haarcascade_frontalface_default.xml"

def getImage(ros_image):
    global image
    image = cv2.cvtColor(bridge.imgmsg_to_cv2(ros_image), cv2.COLOR_RGB2BGR)
def getDepthImage(ros_depth_image):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(ros_depth_image, ros_depth_image.encoding)
def getRay(point):
    global ray
    ray = numpy.array(camera_model.projectPixelTo3dRay(point))
def getCameraInfo(camera_info):
    global cameraInfo
    cameraInfo = camera_info

def draw_faces(image, faces):
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
def detect_faces(image, face_classifier, scale_factor, min_neighbors, localize=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if localize:
        draw_faces(image, faces)
    return faces

if __name__ == "__main__":
    rospy.init_node("intel_realsense_face_detection_and_ranging")
    rate = rospy.Rate(10)
    image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, getImage)
    image_depth_subscriber = rospy.Subscriber("/camera/depth/image_rect_raw", Image, getDepthImage)
    camera_info_subscriber = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, getCameraInfo)
    while cameraInfo == None:
        rate.sleep()
    camera_model.fromCameraInfo(cameraInfo)
    while image == None:
        rate.sleep()
    face_classifier = cv2.CascadeClassifier(cascade_file)
    while not rospy.is_shutdown() and cv2.waitKey(1) != ord('q'):
        faces = detect_faces(image, face_classifier, scale_factor, min_neighbors, localize=True)
        cv2.imshow("Image", image)
    cv2.destroyAllWindows()