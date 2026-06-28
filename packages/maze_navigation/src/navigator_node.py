#!/usr/bin/env python3
"""
navigator_node.py — Centrale state machine voor de Duckiebot maze navigatie.

States:
  LANE_FOLLOWING   — bot rijdt, lane controller actief
  OBSTACLE_STOPPED — duckie gedetecteerd (lane controller stopt al zelf via
                     road_anomaly_watcher), navigator toont dit in dashboard
  GOAL_REACHED     — E bereikt op basis van encoder afstand, bot stopt permanent

Afstandsmeting:
  Gebruikt wiel-encoder ticks (links + rechts gemiddeld).
  dist_per_tick = (2 * pi * WHEEL_RADIUS) / ENCODER_RESOLUTION
  DB21 Duckiebot: radius=0.0318m, resolution=135 ticks/omwenteling
"""

import os
import math
import rospy
import yaml

from std_msgs.msg import Int32MultiArray, Int32, String, Empty
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, BoolStamped, StopLineReading, FSMState


NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

# Duckiebot DB21 wiel parameters
WHEEL_RADIUS_M   = 0.0318
ENCODER_TICKS    = 135   # ticks per volledige omwenteling


class NavigatorNode:
    def __init__(self):
        rospy.init_node('navigator_node')

        veh = os.environ.get('VEHICLE_NAME', rospy.get_param('~veh', 'duckiebot21'))

        # Parameters
        self.goal_distance_m = rospy.get_param('~goal_distance_m', 2.40)
        self.debounce_frames = rospy.get_param('~debounce_frames', 10)
        self.tile_size_m     = rospy.get_param('~tile_size_m', 0.50)
        self.stop_hz         = rospy.get_param('~stop_hz', 20.0)

        config_file = rospy.get_param('~config', '')
        self.graph = None
        if config_file:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
            self.graph = cfg['graph']

        # Afstand per tick (beide wielen gelijk aangenomen)
        self.dist_per_tick = (2.0 * math.pi * WHEEL_RADIUS_M) / ENCODER_TICKS

        # Encoder toestand
        self.left_ticks_start  = None
        self.right_ticks_start = None
        self.left_ticks        = 0
        self.right_ticks       = 0

        # Navigatie toestand
        self.state            = 'LANE_FOLLOWING'
        self.distance_m       = 0.0
        self.obstacle_flag    = False
        self.clear_count      = 0
        self.path             = []
        self.current_node_idx = 0

        # --- Subscribers ---
        rospy.Subscriber(
            f'/{veh}/left_wheel_encoder_driver_node/tick',
            WheelEncoderStamped,
            self.cb_left_encoder,
            queue_size=1
        )
        rospy.Subscriber(
            f'/{veh}/right_wheel_encoder_driver_node/tick',
            WheelEncoderStamped,
            self.cb_right_encoder,
            queue_size=1
        )
        rospy.Subscriber(
            f'/{veh}/object_detection_node/obstacle_distance',
            StopLineReading,
            self.cb_obstacle,
            queue_size=1
        )
        rospy.Subscriber(
            '/planned_path',
            Int32MultiArray,
            self.cb_path,
            queue_size=1
        )
        rospy.Subscriber(
            '/reset_navigation',
            Empty,
            self.cb_reset,
            queue_size=1
        )
        rospy.Subscriber(
            f'/{veh}/fsm_node/mode',
            FSMState,
            self.cb_fsm,
            queue_size=1
        )

        # --- Publishers ---
        self.pub_wheels = rospy.Publisher(
            f'/{veh}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1
        )
        self.pub_lane_switch = rospy.Publisher(
            f'/{veh}/lane_controller_node/switch',
            BoolStamped,
            queue_size=1
        )
        self.pub_current_node = rospy.Publisher('/current_node', Int32, queue_size=1)
        self.pub_state        = rospy.Publisher('/navigator_state', String, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.stop_hz), self.cb_timer)

        rospy.loginfo(f"[navigator] Gestart | veh={veh} | doel={self.goal_distance_m:.2f}m")
        rospy.loginfo(f"[navigator] dist_per_tick={self.dist_per_tick*1000:.2f}mm")

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def cb_path(self, msg):
        self.path = list(msg.data)
        self.current_node_idx = self.path[0] if self.path else 0
        names = [NODE_NAMES[i] for i in self.path]
        rospy.loginfo(f"[navigator] Pad ontvangen: {' -> '.join(names)}")

    def cb_reset(self, _msg):
        self._reset()

    def cb_fsm(self, msg):
        if msg.state == 'LANE_FOLLOWING' and self.state == 'GOAL_REACHED':
            self._reset()

    def cb_left_encoder(self, msg):
        if self.left_ticks_start is None:
            self.left_ticks_start = msg.data
        self.left_ticks = msg.data - self.left_ticks_start
        self._update_distance()

    def cb_right_encoder(self, msg):
        if self.right_ticks_start is None:
            self.right_ticks_start = msg.data
        self.right_ticks = msg.data - self.right_ticks_start
        self._update_distance()

    def cb_obstacle(self, msg):
        if msg.stop_line_detected:
            self.clear_count = 0
            self.obstacle_flag = True
            if self.state == 'LANE_FOLLOWING':
                self._transition('OBSTACLE_STOPPED')
        else:
            if self.obstacle_flag:
                self.clear_count += 1
                if self.clear_count >= self.debounce_frames:
                    self.obstacle_flag = False
                    self.clear_count = 0
                    if self.state == 'OBSTACLE_STOPPED':
                        self._transition('LANE_FOLLOWING')

    def cb_timer(self, _event):
        if self.state in ('GOAL_REACHED', 'OBSTACLE_STOPPED'):
            self._send_stop()

        # Publiceer status voor dashboard
        self.pub_current_node.publish(Int32(data=self.current_node_idx))

        remaining  = max(0.0, self.goal_distance_m - self.distance_m)
        node_name  = NODE_NAMES[self.current_node_idx] if self.current_node_idx < len(NODE_NAMES) else "?"
        status = (
            f"State={self.state} | "
            f"Node={node_name} | "
            f"Dist={self.distance_m:.2f}m | "
            f"Rest={remaining:.2f}m"
        )
        self.pub_state.publish(String(data=status))

    # ------------------------------------------------------------------ #
    # Hulpmethoden                                                         #
    # ------------------------------------------------------------------ #

    def _reset(self):
        self.left_ticks_start  = None
        self.right_ticks_start = None
        self.left_ticks        = 0
        self.right_ticks       = 0
        self.distance_m        = 0.0
        self.obstacle_flag     = False
        self.clear_count       = 0
        self.current_node_idx  = self.path[0] if self.path else 0
        self.state             = 'LANE_FOLLOWING'
        self._set_lane_controller(True)
        rospy.loginfo("[navigator] Reset uitgevoerd — opnieuw A->E")

    def _update_distance(self):
        if self.state != 'LANE_FOLLOWING':
            return

        avg_ticks = (abs(self.left_ticks) + abs(self.right_ticks)) / 2.0
        self.distance_m = avg_ticks * self.dist_per_tick

        self._update_current_node()

        if self.distance_m >= self.goal_distance_m:
            self._transition('GOAL_REACHED')

    def _transition(self, new_state):
        if new_state == self.state:
            return
        rospy.loginfo(f"[navigator] {self.state} -> {new_state}")
        prev_state = self.state
        self.state = new_state

        if new_state == 'OBSTACLE_STOPPED':
            self._set_lane_controller(False)
            self._send_stop()
            rospy.logwarn(f"[navigator] GESTOPT — duckie obstakel gedetecteerd op {self.distance_m:.2f}m | verwijder de duckie om verder te rijden")
        elif new_state == 'LANE_FOLLOWING' and prev_state == 'OBSTACLE_STOPPED':
            self._set_lane_controller(True)
            rospy.loginfo(f"[navigator] Obstakel verwijderd — verder rijden naar doel ({max(0.0, self.goal_distance_m - self.distance_m):.2f}m resterend)")
        elif new_state == 'GOAL_REACHED':
            self._set_lane_controller(False)
            self._send_stop()
            rospy.loginfo(f"[navigator] Doel bereikt na {self.distance_m:.2f}m!")

    def _set_lane_controller(self, active: bool):
        msg = BoolStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = active
        self.pub_lane_switch.publish(msg)
        rospy.loginfo(f"[navigator] Lane controller: {'AAN' if active else 'UIT'}")

    def _send_stop(self):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left  = 0.0
        msg.vel_right = 0.0
        self.pub_wheels.publish(msg)

    def _update_current_node(self):
        if not self.path or self.graph is None:
            return
        cumulative = 0.0
        for i in range(len(self.path) - 1):
            from_node = self.path[i]
            to_node   = self.path[i + 1]
            seg_tiles = next(
                (w for (nb, w) in self.graph[from_node] if nb == to_node), 1
            )
            cumulative += seg_tiles * self.tile_size_m
            if self.distance_m < cumulative:
                self.current_node_idx = from_node
                return
        self.current_node_idx = self.path[-1]


if __name__ == '__main__':
    node = NavigatorNode()
    rospy.spin()
