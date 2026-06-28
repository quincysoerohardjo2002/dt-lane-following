#!/usr/bin/env python3
import heapq
import rospy
import yaml
from std_msgs.msg import Int32MultiArray

def dijkstra(graph, start, goal):
    """
    Kortste pad via Dijkstra. Bij gelijke kosten wint het pad via lagere node-indices
    (geeft A->B->E boven A->C->D->E).
    Returns: (kosten, pad als lijst van indices) of (inf, []) als geen pad.
    """
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    prev = [-1] * n
    # heap: (kosten, node_index) — node_index als tie-breaker geeft lagere indices voorrang
    heap = [(0, start)]

    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]:
            continue
        if u == goal:
            break
        for neighbor, weight in graph[u]:
            new_cost = cost + weight
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev[neighbor] = u
                heapq.heappush(heap, (new_cost, neighbor))

    if dist[goal] == float('inf'):
        return float('inf'), []

    path = []
    node = goal
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()
    return dist[goal], path


class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node')

        config_file = rospy.get_param('~config', '')
        if not config_file:
            rospy.logerr("Geen config bestand opgegeven (~config parameter)")
            rospy.signal_shutdown("Geen config")
            return

        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        self.graph = cfg['graph']
        self.node_names = cfg['node_names']
        self.start = cfg['start_node']
        self.goal = cfg['goal_node']
        self.tile_size = cfg['tile_size_m']

        self.pub = rospy.Publisher('/planned_path', Int32MultiArray, queue_size=1, latch=True)

        cost, path = dijkstra(self.graph, self.start, self.goal)

        if not path:
            rospy.logerr(f"Geen pad gevonden van {self.node_names[self.start]} naar {self.node_names[self.goal]}")
            rospy.signal_shutdown("Geen pad")
            return

        names = [self.node_names[i] for i in path]
        rospy.loginfo(f"Kortste pad: {' -> '.join(names)} (kosten: {cost} tegels = {cost * self.tile_size:.2f}m)")

        msg = Int32MultiArray()
        msg.data = path
        self.pub.publish(msg)
        rospy.loginfo(f"Pad gepubliceerd op /planned_path: {path}")


if __name__ == '__main__':
    node = PathPlannerNode()
    rospy.spin()
