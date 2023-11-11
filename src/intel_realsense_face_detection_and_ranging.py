#!/usr/bin/env python3

import rospy, cv2, numpy, image_geometry
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

bridge = CvBridge()
camera_model = image_geometry.PinholeCameraModel()

cameraInfo = None
image = None
depth_image = None

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

scale_factor = 1.3
min_neighbors = 5
cascade_file = "xml/haarcascade_frontalface_default.xml"

def getCameraInfo(camera_info):
    global cameraInfo
    cameraInfo = camera_info
def getImage(ros_image):
    global image
    image = cv2.cvtColor(bridge.imgmsg_to_cv2(ros_image), cv2.COLOR_RGB2BGR)
def getDepthImage(ros_depth_image):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(ros_depth_image, ros_depth_image.encoding)
def getRay(point):
    return numpy.array(camera_model.projectPixelTo3dRay(point))

def measureFaceDepth(face):
    x, y, w, h = face
    face_depth = depth_image[x:x+w, y:y+h].flatten()
    face_depth = face_depth[face_depth != 0]
    return numpy.sum(face_depth)/len(face_depth)
def draw_faces(image, faces, ranging=True):
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        if ranging == True:
            ray = getRay([x+w//2, y+h//2])
            depth = measureFaceDepth(face)
            coordinates = ray * depth
            cv2.putText(image, f"({coordinates[0]:.2f},{coordinates[1]:.2f},{coordinates[2]:.2f})mm", (x, y+h), cv2.FONT_HERSHEY_COMPLEX, 1, WHITE, 2)
def detect_faces(image, face_classifier, scale_factor, min_neighbors, localize=True, ranging=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if localize:
        draw_faces(image, faces, ranging=ranging)
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
    while True:
        try:
            cv2.imshow("Face Detection and Ranging", image)
            break
        except:
            pass
    face_classifier = cv2.CascadeClassifier(cascade_file)
    while not rospy.is_shutdown() and cv2.waitKey(1) != ord('q'):
        faces = detect_faces(image, face_classifier, scale_factor, min_neighbors, localize=True, ranging=True)
        cv2.imshow("Face Detection and Ranging", image)
    cv2.destroyAllWindows()