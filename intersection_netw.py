"""
Custom unsignalized 4-way intersection network for Flow -- parameterized
so that ONE class can generate all 9 scenario variants shown in Fig. 2 of:

Kai et al., "A Multi-Task Reinforcement Learning Approach for Navigating
Unsignalized Intersections", IV 2020.

Rather than writing 9 separate Network subclasses, the geometry that
varies across the paper's 9 scenarios -- lane count per arm, and whether
an arm has a dedicated turn pocket -- is expressed as `additional_params`
so a single class + 9 config dicts (see `SCENARIO_CONFIGS` at bottom)
reproduces all 9 maps.

Layout (compass directions), each arm has APPROACH_LENGTH meters before
the junction and the same length after it:

                    north_in
                        |
                        v
    west_in  --->  [ CENTER ]  <---  east_in
                        ^
                        |
                    south_in  (etc., "_out" edges mirror this)

Every arm has an "in" edge feeding the junction and an "out" edge leaving
it, so the ego vehicle's route is: <arm>_in -> center -> <opposite/left/
right arm>_out depending on the task (straight / left / right).

NOTE: this file is written against Flow's Network base-class API
(specify_nodes / specify_edges / specify_connections / specify_routes /
specify_types). It has not been executed against a live Flow+SUMO
install in this environment -- please run a quick sanity simulation
(see bottom of this file) before trusting it in training.
"""

from flow.networks.base import Network

ARMS = ["north", "south", "east", "west"]

# Each arm's lane layout is described by:
#   "through_lanes": lanes shared by straight/left/right traffic
#   "turn_pocket": None, or a dict {"movement": "left"/"right",
#                                    "lanes": n} for a dedicated pocket
#
# `lanes` on an "_in" edge = through_lanes + (pocket lanes if any).
# Pocket lanes are always the outermost lane(s) (rightmost for a right
# pocket, leftmost for a left pocket), consistent with real-world design
# and with SUMO's right-aligned lane numbering (lane 0 = rightmost).
ADDITIONAL_NET_PARAMS = {
    # length (m) of each approach/exit arm
    "approach_length": 100,
    # speed limit (m/s) on the arms
    "speed_limit": 18,
    # per-arm lane geometry, keyed by "north"/"south"/"east"/"west"
    "arm_config": {
        arm: {"through_lanes": 1, "turn_pocket": None} for arm in ARMS
    },
}


class UnsignalizedIntersectionNetwork(Network):
    """A single unsignalized 4-way intersection with N/S/E/W arms.

    Each arm contributes one "in" edge (approaching the junction) and one
    "out" edge (leaving the junction). Lane count and turn-pocket layout
    per arm are configurable via `net_params.additional_params['arm_config']`,
    which is what lets one class reproduce all 9 Fig. 2 scenarios.
    """

    def __init__(self, name, vehicles, net_params, initial_config=None,
                 traffic_lights=None):
        for p in ["approach_length", "speed_limit", "arm_config"]:
            if p not in net_params.additional_params:
                raise KeyError(f"Network parameter '{p}' not supplied")
        arm_cfg = net_params.additional_params["arm_config"]
        for arm in ARMS:
            if arm not in arm_cfg:
                raise KeyError(f"arm_config missing entry for '{arm}'")
        super().__init__(name, vehicles, net_params, initial_config,
                          traffic_lights)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _in_lanes(arm_cfg_entry):
        """Total lanes on the '_in' edge for one arm (through + pocket)."""
        n = arm_cfg_entry["through_lanes"]
        if arm_cfg_entry.get("turn_pocket"):
            n += arm_cfg_entry["turn_pocket"]["lanes"]
        return n

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def specify_nodes(self, net_params):
        L = net_params.additional_params["approach_length"]

        nodes = [
            {"id": "center", "x": 0, "y": 0, "radius": 8,
             "type": "priority"},
            {"id": "north_end", "x": 0, "y": L},
            {"id": "south_end", "x": 0, "y": -L},
            {"id": "east_end", "x": L, "y": 0},
            {"id": "west_end", "x": -L, "y": 0},
        ]
        return nodes

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def specify_edges(self, net_params):
        L = net_params.additional_params["approach_length"]
        arm_cfg = net_params.additional_params["arm_config"]

        end_node = {
            "north": "north_end", "south": "south_end",
            "east": "east_end", "west": "west_end",
        }

        edges = []
        for arm in ARMS:
            n_in = self._in_lanes(arm_cfg[arm])
            # out-edges: keep them simple (through_lanes only -- pockets
            # only matter for the approach, not the exit)
            n_out = arm_cfg[arm]["through_lanes"]

            edges.append({
                "id": f"{arm}_in", "type": f"{arm}_in_type",
                "from": end_node[arm], "to": "center", "length": L,
                "numLanes": n_in,
            })
            edges.append({
                "id": f"{arm}_out", "type": f"{arm}_out_type",
                "from": "center", "to": end_node[arm], "length": L,
                "numLanes": n_out,
            })
        return edges

    def specify_types(self, net_params):
        speed = net_params.additional_params["speed_limit"]
        arm_cfg = net_params.additional_params["arm_config"]

        types = []
        for arm in ARMS:
            n_in = self._in_lanes(arm_cfg[arm])
            n_out = arm_cfg[arm]["through_lanes"]
            types.append({"id": f"{arm}_in_type", "numLanes": n_in,
                           "speed": speed})
            types.append({"id": f"{arm}_out_type", "numLanes": n_out,
                           "speed": speed})
        return types

    # ------------------------------------------------------------------
    # Connections (who may turn where at the junction)
    # ------------------------------------------------------------------
    
# node_calibration_hint_0 = (0, 0)
# node_calibration_hint_1 = (10, 10)
# node_calibration_hint_2 = (20, 20)
# node_calibration_hint_3 = (30, 30)
# node_calibration_hint_4 = (40, 40)
# node_calibration_hint_5 = (50, 50)
# node_calibration_hint_6 = (60, 60)
# node_calibration_hint_7 = (70, 70)
# node_calibration_hint_8 = (80, 80)
# node_calibration_hint_9 = (90, 90)
# node_calibration_hint_10 = (100, 100)
# node_calibration_hint_11 = (110, 110)
# node_calibration_hint_12 = (120, 120)
# node_calibration_hint_13 = (130, 130)
# node_calibration_hint_14 = (140, 140)
# node_calibration_hint_15 = (150, 150)
# node_calibration_hint_16 = (160, 160)
# node_calibration_hint_17 = (170, 170)
# node_calibration_hint_18 = (180, 180)
# node_calibration_hint_19 = (190, 190)
# node_calibration_hint_20 = (200, 200)
# node_calibration_hint_21 = (210, 210)
# node_calibration_hint_22 = (220, 220)
# node_calibration_hint_23 = (230, 230)
# node_calibration_hint_24 = (240, 240)