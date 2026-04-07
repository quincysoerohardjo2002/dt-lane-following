#!/usr/bin/env python3

import os
import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped, StopLineReading
from geometry_msgs.msg import Point

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.camera.homography import HomographyToolkit

from duckietown.dtros import DTROS, NodeType, TopicType

from object_detection_model.model import MLModel
from object_detection_model.config import FORWARD_PWM, CONF_THRESHOLD


class ObjectDetectionNode(DTROS):
    """
    Performs ONNX-based object detection on camera images to detect duckie obstacles.

    Can operate in two modes:
      - **Standalone**: publishes PWM wheel commands directly (go straight / stop).
      - **Combined with lane following**: publishes a StopLineReading on
        ``~obstacle_distance`` so that the lane controller can handle stopping.

    Subscribers:
        ~image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Camera image
        ~camera_info (:obj:`sensor_msgs.msg.CameraInfo`): Camera intrinsics

    Publishers:
        ~wheels_cmd (:obj:`duckietown_msgs.msg.WheelsCmdStamped`): Direct wheel commands (standalone)
        ~obstacle_distance (:obj:`duckietown_msgs.msg.StopLineReading`): Obstacle info for lane controller
        ~detection_image/compressed (:obj:`sensor_msgs.msg.CompressedImage`): Annotated debug image
    """

    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION,
            fsm_controlled=False,
        )

        # parameters
        self._publish_wheels = rospy.get_param("~publish_wheels", True)

        # state
        self.bridge = CvBridge()
        self.camera = None
        self.projector = None
        self.camera_info_received = False
        self.model = None
        self._first_detection_done = False

        # initialize model
        try:
            self.model = MLModel()
            self.log("MLModel initialized successfully")
        except Exception as e:
            self.logerr(f"Failed to initialize MLModel: {e}")

        # subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.cb_image,
            buff_size=10000000,
            queue_size=1,
        )
        self.sub_camera_info = rospy.Subscriber(
            "~camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1,
        )

        # publishers
        self.pub_wheels_cmd = rospy.Publisher(
            "~wheels_cmd",
            WheelsCmdStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL,
        )
        self.pub_obstacle_distance = rospy.Publisher(
            "~obstacle_distance",
            StopLineReading,
            queue_size=1,
        )
        self.pub_detection_image = rospy.Publisher(
            "~detection_image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.log("Initialized!")

    def cb_camera_info(self, msg):
        """Sets up CameraModel and GroundProjector from camera calibration."""
        if self.camera_info_received:
            return

        self.log("Received camera info")

        self.camera = CameraModel(
            width=msg.width,
            height=msg.height,
            K=np.reshape(msg.K, (3, 3)),
            D=np.reshape(msg.D, (5,)),
            P=np.reshape(msg.P, (3, 4)),
        )

        homography = self._load_extrinsics()
        if homography is not None:
            self.camera.H = homography
            self.projector = GroundProjector(self.camera)
            if self.model is not None:
                self.model.set_ground_projector(self.projector)
            self.log("GroundProjector initialized for object detection")
        else:
            self.logwarn("No homography loaded; distance estimation disabled")

        self.camera_info_received = True

    def cb_image(self, msg):
        """Processes each camera image through the object detection model."""
        if self.model is None:
            return

        # decode compressed image
        try:
            img_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.logerr(f"Image decode error: {e}")
            return

        # resize to model input size
        img_resized = cv2.resize(img_bgr, (self.model.net_w, self.model.net_h))

        # run detection
        detections, should_stop, had_error = self.model.get_wheel_velocities_from_image(img_resized)

        if had_error:
            return

        if not self._first_detection_done:
            self.log("First detection processed")
            self._first_detection_done = True

        # publish obstacle distance for lane controller integration
        self._publish_obstacle_reading(detections, should_stop, msg.header)

        # publish direct wheel commands (standalone mode)
        if self._publish_wheels:
            self._publish_wheel_cmd(should_stop, msg.header)

        # publish annotated debug image
        if self.pub_detection_image.get_num_connections() > 0:
            self._publish_debug_image(img_resized, detections, msg.header)

    def _publish_wheel_cmd(self, should_stop, header):
        """Publishes direct PWM wheel commands."""
        wheel_msg = WheelsCmdStamped()
        wheel_msg.header = header
        if should_stop:
            wheel_msg.vel_left = 0.0
            wheel_msg.vel_right = 0.0
        else:
            wheel_msg.vel_left = FORWARD_PWM
            wheel_msg.vel_right = FORWARD_PWM
        self.pub_wheels_cmd.publish(wheel_msg)

    def _publish_obstacle_reading(self, detections, should_stop, header):
        """Publishes a StopLineReading message for the lane controller."""
        msg = StopLineReading()
        msg.header = header
        msg.stop_line_detected = should_stop
        msg.at_stop_line = should_stop
        if should_stop:
            msg.stop_pose.x = 0.15
            msg.stop_pose.y = 0.0
        self.pub_obstacle_distance.publish(msg)

    def _publish_debug_image(self, img, detections, header):
        """Publishes an annotated image showing bounding boxes."""
        if detections is None:
            return
        vis = img.copy()
        for x1, y1, x2, y2, score, cls in detections:
            if score < CONF_THRESHOLD:
                continue
            cv2.rectangle(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"{score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        debug_msg = self.bridge.cv2_to_compressed_imgmsg(vis)
        debug_msg.header = header
        self.pub_detection_image.publish(debug_msg)

    def _load_extrinsics(self):
        """Loads the homography matrix from the extrinsic calibration file."""
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        if not os.path.isfile(cali_file):
            self.logwarn(
                f"Can't find calibration file: {cali_file}\n Using default calibration instead."
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        if not os.path.isfile(cali_file):
            self.logerr("Found no calibration file ... distance estimation disabled")
            return None

        try:
            H = HomographyToolkit.load_from_disk(cali_file, return_date=False)
            return H.reshape((3, 3))
        except Exception as e:
            self.logerr(f"Error in parsing calibration file {cali_file}: {e}")
            return None


if __name__ == "__main__":
    node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
