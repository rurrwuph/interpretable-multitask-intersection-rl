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


