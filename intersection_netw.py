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
    def specify_connections(self, net_params):
        """Build lane-level connections per arm, respecting turn pockets.

        Movement geometry (compass-relative to the *_in edge's arm):
          north_in -> south_out (straight), east_out (right), west_out (left)
          south_in -> north_out (straight), west_out (right), east_out (left)
          east_in  -> west_out  (straight), south_out (right), north_out (left)
          west_in  -> east_out  (straight), north_out (right), south_out (left)

        Lane numbering follows SUMO convention: lane 0 = rightmost.
        If an arm has a turn pocket, that pocket occupies the outermost
        lane(s) on the correct side (lane 0 upward for a right pocket,
        the highest-numbered lane(s) for a left pocket); through lanes
        carry straight movement and, if no pocket is defined for a given
        turn, that turn too (unrestricted).
        """
        arm_cfg = net_params.additional_params["arm_config"]

        movement_map = {
            # A vehicle entering from "north" travels southbound (toward
            # center from y=+L to y=0). Facing south: straight=south_out,
            # right turn (west side, since south-facing right hand points
            # west)=west_out, left=east_out.
            "north": {"straight": "south_out", "right": "west_out",
                      "left": "east_out"},
            # Entering from "south", traveling northbound. Facing north:
            # straight=north_out, right=east_out, left=west_out.
            "south": {"straight": "north_out", "right": "east_out",
                      "left": "west_out"},
            # Entering from "east", traveling westbound. Facing west:
            # straight=west_out, right=north_out, left=south_out.
            "east": {"straight": "west_out", "right": "north_out",
                     "left": "south_out"},
            # Entering from "west", traveling eastbound. Facing east:
            # straight=east_out, right=south_out, left=north_out.
            "west": {"straight": "east_out", "right": "south_out",
                     "left": "north_out"},
        }

        connections = {}
        for arm in ARMS:
            cfg = arm_cfg[arm]
            through = cfg["through_lanes"]
            pocket = cfg.get("turn_pocket")
            from_edge = f"{arm}_in"
            moves = movement_map[arm]

            # number of lanes on each possible destination "_out" edge,
            # needed so we don't always dump every connection onto
            # toLane 0 (which leaves lane 1+ on multi-lane exit edges
            # unreachable -- "Lane 'X_out_1' is not connected from any
            # incoming edge" warnings). Destination edges always have
            # `through_lanes` lanes (see specify_edges: n_out uses
            # through_lanes only, pockets don't apply to exits).
            def n_lanes_on(dest_edge):
                dest_arm = dest_edge.replace("_out", "")
                return arm_cfg[dest_arm]["through_lanes"]

            conns = []

            if pocket is None:
                # No dedicated pocket: every through lane permits all
                # three movements (straight/left/right), matching a
                # simple single/shared-lane approach. Spread fromLanes
                # round-robin across the destination edge's lanes so
                # every destination lane actually receives a connection.
                for lane in range(through):
                    for mv in ("straight", "left", "right"):
                        dest_edge = moves[mv]
                        n_dest_lanes = n_lanes_on(dest_edge)
                        to_lane = lane % n_dest_lanes
                        conns.append({
                            "from": from_edge, "to": dest_edge,
                            "fromLane": lane, "toLane": to_lane,
                        })
            else:
                pocket_mv = pocket["movement"]  # "left" or "right"
                pocket_lanes = pocket["lanes"]
                # Pocket lanes: "right" pocket -> lowest lane numbers
                # (0..pocket_lanes-1); "left" pocket -> highest lane
                # numbers (through..through+pocket_lanes-1).
                if pocket_mv == "right":
                    pocket_lane_ids = list(range(pocket_lanes))
                    through_lane_ids = list(
                        range(pocket_lanes, pocket_lanes + through))
                else:  # "left"
                    through_lane_ids = list(range(through))
                    pocket_lane_ids = list(
                        range(through, through + pocket_lanes))

                pocket_dest = moves[pocket_mv]
                n_pocket_dest_lanes = n_lanes_on(pocket_dest)
                for i, lane in enumerate(pocket_lane_ids):
                    conns.append({
                        "from": from_edge, "to": pocket_dest,
                        "fromLane": lane,
                        "toLane": i % n_pocket_dest_lanes,
                    })

                # through lanes carry straight + whichever turn has no
                # dedicated pocket
                other_mv = "left" if pocket_mv == "right" else "right"
                straight_dest = moves["straight"]
                other_dest = moves[other_mv]
                n_straight_lanes = n_lanes_on(straight_dest)
                n_other_lanes = n_lanes_on(other_dest)
                for i, lane in enumerate(through_lane_ids):
                    conns.append({
                        "from": from_edge, "to": straight_dest,
                        "fromLane": lane,
                        "toLane": i % n_straight_lanes,
                    })
                    conns.append({
                        "from": from_edge, "to": other_dest,
                        "fromLane": lane,
                        "toLane": i % n_other_lanes,
                    })

            connections[from_edge] = conns

        return connections

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def specify_routes(self, net_params):
        # Only "_in" edges need route entries. Per Flow's own docs
        # (flow/networks/base.py): if a starting edge has no entry in
        # specify_routes(), a vehicle on it simply exits the network
        # once it reaches the end of that edge -- which is exactly the
        # correct behavior for "_out" edges, so they intentionally have
        # NO entries here. (An earlier version of this file added
        # single-edge entries for "_out" edges too, chasing a
        # KeyError -- that error's real cause was the ego vehicle not
        # being removed before flow_env.reset() tried to respawn it
        # from a stale "_out" edge; see the fix in intersection_env.py.
        # Do not re-add "_out" entries here.)
        routes = {
            "north_in": ["north_in", "south_out"],
            "south_in": ["south_in", "north_out"],
            "east_in": ["east_in", "west_out"],
            "west_in": ["west_in", "east_out"],
        }
        return routes

    


# Convenience: named routes for the three ego tasks, keyed by entry arm.
# Use these in the environment when constructing the ego vehicle's route
# based on the sampled task vector g.
EGO_ROUTES = {
    # west_in: vehicle travels eastbound. Facing east: right=south,
    # left=north (matches corrected movement_map in specify_connections).
    "west_in": {
        "straight": ["west_in", "east_out"],
        "left": ["west_in", "north_out"],
        "right": ["west_in", "south_out"],
    },
    # east_in: vehicle travels westbound. Facing west: right=north,
    # left=south.
    "east_in": {
        "straight": ["east_in", "west_out"],
        "left": ["east_in", "south_out"],
        "right": ["east_in", "north_out"],
    },
    # north_in: vehicle travels southbound. Facing south: right=west,
    # left=east.
    "north_in": {
        "straight": ["north_in", "south_out"],
        "left": ["north_in", "east_out"],
        "right": ["north_in", "west_out"],
    },
    # south_in: vehicle travels northbound. Facing north: right=east,
    # left=west.
    "south_in": {
        "straight": ["south_in", "north_out"],
        "left": ["south_in", "west_out"],
        "right": ["south_in", "east_out"],
    },
}


# ----------------------------------------------------------------------
# Nine scenario configs, approximating the lane-count / turn-pocket
# variety visible across panels (a)-(i) of Fig. 2. These are a best-
# effort reading of the figure (exact SUMO source files for the paper's
# 9 maps aren't public), grouped roughly by rising complexity:
#   (a)-(c): simple crossings, 1-2 through lanes, no dedicated pockets
#   (d)-(f): mixed lane counts, some with a dedicated left- or
#            right-turn pocket on the busier arms
#   (g)-(i): higher lane count / asymmetric approaches (e.g. one axis
#            wider than the other), still no signal control
#
# Treat these as a reasonable starting point to match "9 different
# intersections with varying geometry," not as a pixel-exact reproduction
# of the authors' unpublished net files. Adjust freely once you compare
# against the figure again or your own judgment of what each panel shows.
# ----------------------------------------------------------------------

def _uniform(through_lanes, pocket=None):
    """Helper: same lane config on all 4 arms."""
    entry = {"through_lanes": through_lanes, "turn_pocket": pocket}
    return {arm: dict(entry) for arm in ARMS}


SCENARIO_CONFIGS = {
    # (a) 2 through lanes all round, no pockets -- widest simple crossing
    "scenario_a": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": _uniform(2)},

    # (b) 1 through lane all round -- simplest possible crossing
    "scenario_b": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": _uniform(1)},

    # (c) 1 through lane, but every arm has a dedicated right-turn pocket
    "scenario_c": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": _uniform(
                       1, pocket={"movement": "right", "lanes": 1})},

    # (d) mixed: N/S arms 2 lanes, E/W arms 1 lane (asymmetric axes)
    "scenario_d": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": {
                       "north": {"through_lanes": 2, "turn_pocket": None},
                       "south": {"through_lanes": 2, "turn_pocket": None},
                       "east": {"through_lanes": 1, "turn_pocket": None},
                       "west": {"through_lanes": 1, "turn_pocket": None},
                   }},

    # (e) 2 through lanes + dedicated left-turn pocket on every arm
    "scenario_e": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": _uniform(
                       2, pocket={"movement": "left", "lanes": 1})},

    # (f) busier: 3 through lanes, no pockets
    "scenario_f": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": _uniform(3)},

    # (g) mixed pockets: N/S get left pockets, E/W get right pockets
    "scenario_g": {"approach_length": 100, "speed_limit": 18,
                   "arm_config": {
                       "north": {"through_lanes": 1,
                                 "turn_pocket": {"movement": "left",
                                                 "lanes": 1}},
                       "south": {"through_lanes": 1,
                                 "turn_pocket": {"movement": "left",
                                                 "lanes": 1}},
                       "east": {"through_lanes": 1,
                                "turn_pocket": {"movement": "right",
                                                "lanes": 1}},
                       "west": {"through_lanes": 1,
                                "turn_pocket": {"movement": "right",
                                                "lanes": 1}},
                   }},

    # (h) 2 through lanes, no pockets, shorter approach (tighter urban feel)
    "scenario_h": {"approach_length": 70, "speed_limit": 15,
                    "arm_config": _uniform(2)},

    # (i) high-capacity: 3 through lanes + right-turn pocket every arm
    "scenario_i": {"approach_length": 120, "speed_limit": 18,
                   "arm_config": _uniform(
                       3, pocket={"movement": "right", "lanes": 1})},
}

def get_valid_spawn_lane(arm_cfg_entry, task_name):
    """
    Returns the appropriate starting lane index for an arm based on
    the task (left, straight, right) and whether a turn pocket exists.
    """
    through_lanes = arm_cfg_entry["through_lanes"]
    pocket = arm_cfg_entry.get("turn_pocket")

    if pocket is None:
        # All movements allowed from any through lane; default to lane 0
        return 0

    pocket_mv = pocket["movement"]  # "left" or "right"
    pocket_lanes = pocket["lanes"]

    if pocket_mv == "right":
        if task_name == "right":
            # Right turn pocket occupies lane 0 .. pocket_lanes - 1
            return 0
        else:
            # Straight and left must use the through lanes (above right pocket)
            return pocket_lanes

    elif pocket_mv == "left":
        if task_name == "left":
            # Left turn pocket occupies outermost upper lanes
            return through_lanes
        else:
            # Straight and right use the lower through lanes
            return 0

    return 0
# pocket_lane_offset_eval_0 = 0.0
# pocket_lane_offset_eval_1 = 0.1
# pocket_lane_offset_eval_2 = 0.2
# pocket_lane_offset_eval_3 = 0.30000000000000004
# pocket_lane_offset_eval_4 = 0.4
# pocket_lane_offset_eval_5 = 0.5
# pocket_lane_offset_eval_6 = 0.6000000000000001
# pocket_lane_offset_eval_7 = 0.7000000000000001
# pocket_lane_offset_eval_8 = 0.8
# pocket_lane_offset_eval_9 = 0.9
# pocket_lane_offset_eval_10 = 1.0
# pocket_lane_offset_eval_11 = 1.1
# pocket_lane_offset_eval_12 = 1.2000000000000002
# pocket_lane_offset_eval_13 = 1.3
# pocket_lane_offset_eval_14 = 1.4000000000000001
# pocket_lane_offset_eval_15 = 1.5
# pocket_lane_offset_eval_16 = 1.6
# pocket_lane_offset_eval_17 = 1.7000000000000002
# pocket_lane_offset_eval_18 = 1.8
# pocket_lane_offset_eval_19 = 1.9000000000000001