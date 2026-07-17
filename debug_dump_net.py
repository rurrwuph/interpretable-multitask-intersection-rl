"""
Debug script: builds the network exactly like training does, but does
NOT run a simulation -- just constructs the Flow env far enough to
generate the .net.xml, then locates and prints it so we can inspect the
ACTUAL compiled connections SUMO/netconvert produced for west_in,
independent of what our Python specify_connections() claims to emit.

Run this on your machine:
    python debug_dump_net.py

It will print the full path to the generated .net.xml and the relevant
<connection> lines for west_in, so we can see exactly what netconvert
kept vs dropped.
"""

import glob
import os
import time

from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams
import numpy as np
from gymnasium import spaces

from intersection_netw import UnsignalizedIntersectionNetwork
from Network_scenario import vehicles, build_net_params, initialConfig, sumoParams, envParams


class _RawFlowEnv(FlowEnv):
    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        pass

    def get_state(self):
        return np.zeros(26, dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        return 0.0


print("Building network...")
network = UnsignalizedIntersectionNetwork(
    name="debug_net_dump",
    vehicles=vehicles,
    net_params=build_net_params("scenario_b"),
    initial_config=initialConfig,
    traffic_lights=TrafficLightParams(),
)

print("Constructing Flow env (this triggers netconvert)...")
env = _RawFlowEnv(
    env_params=envParams,
    sim_params=sumoParams,
    network=network,
)

# Flow stores the working directory it wrote net files to. Try the
# common attribute names across Flow versions; print whichever exists.
candidates = []
for attr_path in [
    "network.net_params.template",
    "k.network.net_path",
    "k.network.cfg_path",
    "k.network.orig_name",
]:
    obj = env
    ok = True
    for part in attr_path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            ok = False
            break
    if ok:
        print(f"{attr_path} = {obj}")
        candidates.append(str(obj))

# Fallback: search common Flow temp locations for recently-modified
# .net.xml files (last 60 seconds), since the exact attribute name
# varies across Flow forks/versions.
print("\nSearching for recently generated .net.xml files (last 120s)...")
search_roots = [
    os.path.expanduser("~"),
    "/tmp",
]
now = time.time()
found = []
for root in search_roots:
    for path in glob.glob(os.path.join(root, "**", "*.net.xml"), recursive=True):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if now - mtime < 120:
            found.append((mtime, path))

found.sort(reverse=True)
if not found:
    print("No recently-generated .net.xml found. Try increasing the 120s "
          "window above, or check flow's own printed 'Error during start' "
          "message (it usually names the exact temp file path).")
else:
    print(f"Found {len(found)} recently-modified .net.xml file(s):")
    for mtime, path in found[:5]:
        print(f"  {path}  (modified {now - mtime:.0f}s ago)")

    newest_path = found[0][1]
    print(f"\n--- connections FROM west_in in {newest_path} ---")
    with open(newest_path) as f:
        for line in f:
            if '<connection' in line and 'from="west_in"' in line:
                print(line.strip())

    print(f"\n--- all <connection> lines mentioning west_in or west_in_0 ---")
    with open(newest_path) as f:
        for line in f:
            if 'west_in' in line and ('<connection' in line or '<lane' in line):
                print(line.strip())

    # --- NEW: dump junction/node coordinates and the full west_in edge
    # block in this SAME run, before the temp file gets cleaned up. ---
    import re

    print(f"\n--- junction coordinates (center, *_end nodes) ---")
    with open(newest_path) as f:
        content = f.read()
    for node_id in ["center", "north_end", "south_end", "east_end", "west_end"]:
        m = re.search(rf'<junction id="{node_id}"[^>]*>', content)
        if m:
            print(m.group(0))
        else:
            print(f"  (junction '{node_id}' not found -- SUMO may have "
                  f"renamed/merged it; see full junction list below)")

    print(f"\n--- ALL <junction> lines (in case node IDs were renamed) ---")
    for m in re.finditer(r'<junction id="[^"]*"[^>]*>', content):
        print(m.group(0))

    print(f"\n--- full <edge id=\"west_in\"> block ---")
    m = re.search(r'<edge id="west_in".*?</edge>', content, re.DOTALL)
    print(m.group(0) if m else "NOT FOUND")

    print(f"\n--- full <edge id=\"west_out\"> block (for comparison) ---")
    m = re.search(r'<edge id="west_out".*?</edge>', content, re.DOTALL)
    print(m.group(0) if m else "NOT FOUND")

print("\nDone. Ctrl+C now if SUMO GUI/instance is still running.")