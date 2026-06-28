#!/usr/bin/env python3
"""
map_visualizer_node.py — ASCII dashboard dat de positie van de Duckiebot
op de maze kaart toont.

Subscribet op:
  /current_node     (Int32)  — huidige node index
  /planned_path     (Int32MultiArray) — geplande route
  /navigator_state  (String) — status tekst

Publiceert:
  /map_display      (String) — ASCII kaart als string (ook gelogd naar console)
"""

import rospy
from std_msgs.msg import Int32, Int32MultiArray, String


# Vaste ASCII kaart layout
# Elke node heeft een (rij, kolom) positie in de kaart
NODE_POSITIONS = {
    0: (4, 1),   # A
    1: (4, 9),   # B
    2: (3, 1),   # C
    3: (3, 4),   # D
    4: (3, 9),   # E  ← DOEL
    5: (2, 1),   # F
    6: (2, 4),   # G
    7: (1, 1),   # H
    8: (1, 9),   # I
}

NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

BASE_MAP = [
    "  H ─────────────────── I  ",
    "  │                     │  ",
    "  F ────── G             │  ",
    "  │                     │  ",
    "  C ──── D ────────── E  │  ",
    "  │                  │  │  ",
    "  A ─────────────── B ──┘  ",
]


def render_map(current_node: int, path: list) -> str:
    """Render ASCII kaart met huidige node en route gemarkeerd."""
    rows = [list(row) for row in BASE_MAP]
    cols = max(len(r) for r in BASE_MAP)
    # Zorg dat alle rijen even lang zijn
    for r in rows:
        while len(r) < cols:
            r.append(' ')

    for idx, (row, col) in NODE_POSITIONS.items():
        if row >= len(rows) or col >= len(rows[row]):
            continue
        name = NODE_NAMES[idx]
        if idx == current_node:
            rows[row][col] = f"[{name}]"[0]  # We overschrijven één karakter
            # Vervang met markering (3 tekens)
            label = f"[{name}]"
        elif idx in path:
            label = f"*{name}*"
        else:
            label = f" {name} "

        # Schrijf label op positie (kan buiten rij vallen, dan skip)
        for k, ch in enumerate(label):
            c = col + k - 1
            if 0 <= c < len(rows[row]):
                rows[row][c] = ch

    return "\n".join("".join(r) for r in rows)


class MapVisualizerNode:
    def __init__(self):
        rospy.init_node('map_visualizer_node')

        self.current_node = 0
        self.path         = []
        self.status       = "Wachten op data..."

        rospy.Subscriber('/current_node',    Int32,           self.cb_node,   queue_size=1)
        rospy.Subscriber('/planned_path',    Int32MultiArray, self.cb_path,   queue_size=1)
        rospy.Subscriber('/navigator_state', String,          self.cb_status, queue_size=1)

        self.pub_display = rospy.Publisher('/map_display', String, queue_size=1)

        # Kaart 2x per seconde vernieuwen
        rospy.Timer(rospy.Duration(0.5), self.cb_render)

        rospy.loginfo("[visualizer] Dashboard gestart")

    def cb_node(self, msg):
        self.current_node = msg.data

    def cb_path(self, msg):
        self.path = list(msg.data)

    def cb_status(self, msg):
        self.status = msg.data

    def cb_render(self, _event):
        kaart = render_map(self.current_node, self.path)
        route_str = " -> ".join(NODE_NAMES[i] for i in self.path) if self.path else "?"

        output = (
            "\n" + "=" * 40 + "\n"
            f"  DUCKIEBOT NAVIGATIE DASHBOARD\n"
            + "=" * 40 + "\n"
            + kaart + "\n"
            + "-" * 40 + "\n"
            f"  Route : {route_str}\n"
            f"  Status: {self.status}\n"
            + "=" * 40
        )

        rospy.loginfo(output)
        self.pub_display.publish(String(data=output))


if __name__ == '__main__':
    node = MapVisualizerNode()
    rospy.spin()
