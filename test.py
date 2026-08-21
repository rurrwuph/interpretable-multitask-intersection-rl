"""
Test script for the multi-task unsignalized intersection reproduction.

Run this on YOUR machine (with flow + SUMO installed) as:
    python test_setup.py

Two tiers of checks:
  TIER 1 - structural / static checks: importability, dimension math,
           lane-index consistency across all 9 scenarios. These do NOT
           require flow/SUMO and will run in any Python env.
  TIER 2 - live SUMO smoke test: actually builds the network, launches
           SUMO, inserts the ego vehicle, steps it through a short
           episode for each task, and checks the env contract (obs
           shape, reward keys, terminal conditions). Requires flow +
           SUMO to be installed and on PATH. Skipped automatically with
           a clear message if flow can't be imported.

If Tier 2 fails, read the printed traceback -- it will usually point at
one of the exact uncertainty flags called out in earlier review (e.g.
kernel_api attribute name, k.vehicle.add() kwarg names, speed_mode
behavior at the junction).
"""

import sys
import traceback

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, PASS, None))
        print(f"[{PASS}] {name}")
    except AssertionError as e:
        results.append((name, FAIL, str(e)))
        print(f"[{FAIL}] {name}: {e}")
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"[{FAIL}] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def skip(name, reason):
    results.append((name, SKIP, reason))
    print(f"[{SKIP}] {name}: {reason}")


# ======================================================================
# TIER 1: structural checks (no flow/SUMO required)
# ======================================================================

def tier1_import_network_module():
    """intersection_netw.py must import without flow installed by
    stubbing flow.networks.base.Network -- confirms no accidental hard
    dependency leaked into pure config/logic code."""
    import types
    flow_mod = types.ModuleType("flow")
    networks_mod = types.ModuleType("flow.networks")
    base_mod = types.ModuleType("flow.networks.base")

    class FakeNetwork:
        def __init__(self, name, vehicles, net_params, initial_config=None,
                     traffic_lights=None):
            pass

    base_mod.Network = FakeNetwork
    sys.modules["flow"] = flow_mod
    sys.modules["flow.networks"] = networks_mod
    sys.modules["flow.networks.base"] = base_mod

    global inet
    import intersection_netw as inet
    assert hasattr(inet, "SCENARIO_CONFIGS")
    assert hasattr(inet, "EGO_ROUTES")
    assert hasattr(inet, "UnsignalizedIntersectionNetwork")


def tier1_nine_scenarios_present():
    expected = {f"scenario_{c}" for c in "abcdefghi"}
    actual = set(inet.SCENARIO_CONFIGS.keys())
    assert expected == actual, f"expected {expected}, got {actual}"


def tier1_scenario_configs_have_required_keys():
    required = {"approach_length", "speed_limit", "arm_config"}
    arms = {"north", "south", "east", "west"}
    for name, cfg in inet.SCENARIO_CONFIGS.items():
        missing = required - set(cfg.keys())
        assert not missing, f"{name} missing keys {missing}"
        arm_cfg = cfg["arm_config"]
        missing_arms = arms - set(arm_cfg.keys())
        assert not missing_arms, f"{name} missing arms {missing_arms}"
        for arm, entry in arm_cfg.items():
            assert "through_lanes" in entry, f"{name}/{arm} no through_lanes"
            assert entry["through_lanes"] >= 1, f"{name}/{arm} bad lane count"


def tier1_connections_reference_valid_lanes():
    """Re-run the lane-index sanity check for all 9 configs (same check
    used during development) to guard against regressions."""

    class FakeNetParams:
        def __init__(self, additional_params):
            self.additional_params = additional_params

    for name, cfg in inet.SCENARIO_CONFIGS.items():
        net = inet.UnsignalizedIntersectionNetwork.__new__(
            inet.UnsignalizedIntersectionNetwork)
        np_ = FakeNetParams(cfg)
        edges = net.specify_edges(np_)
        types_ = net.specify_types(np_)
        conns = net.specify_connections(np_)

        lane_counts = {t["id"]: t["numLanes"] for t in types_}
        edge_type = {e["id"]: e["type"] for e in edges}

        for from_edge, conn_list in conns.items():
            n_lanes = lane_counts[edge_type[from_edge]]
            for c in conn_list:
                assert 0 <= c["fromLane"] < n_lanes, (
                    f"{name}: bad fromLane {c['fromLane']} on {from_edge} "
                    f"(has {n_lanes} lanes)")


def tier1_ego_routes_cover_all_arms_and_tasks():
    arms = {"north_in", "south_in", "east_in", "west_in"}
    tasks = {"straight", "left", "right"}
    assert set(inet.EGO_ROUTES.keys()) == arms
    for arm, task_routes in inet.EGO_ROUTES.items():
        assert set(task_routes.keys()) == tasks, (
            f"{arm} missing tasks: {tasks - set(task_routes.keys())}")
        for task, route in task_routes.items():
            assert len(route) == 2, f"{arm}/{task} route should be [in, out]"
            assert route[0] == arm, f"{arm}/{task} route doesn't start at {arm}"
            assert route[1].endswith("_out"), (
                f"{arm}/{task} route doesn't end on an _out edge")


def tier1_ego_routes_match_real_connections():
    """Cross-check every EGO_ROUTES (arm, task) -> route against the
    ACTUAL connections specify_connections() generates. This is the
    check that would have caught the left/right movement_map vs
    EGO_ROUTES geometry mismatch bug (right/left swapped relative to
    true compass direction of travel) immediately -- previously the
    lane-index check alone passed even with completely wrong turn
    directions, since it only validated lane numbers, not which edge
    pairs are actually connected."""

    class FakeNetParams:
        def __init__(self, additional_params):
            self.additional_params = additional_params

    # geometry only depends on arm_config having valid structure, not on
    # lane counts -- any scenario works, use the simplest one
    cfg = inet.SCENARIO_CONFIGS["scenario_b"]
    net = inet.UnsignalizedIntersectionNetwork.__new__(
        inet.UnsignalizedIntersectionNetwork)
    conns = net.specify_connections(FakeNetParams(cfg))

    existing_pairs = set()
    for from_edge, conn_list in conns.items():
        for c in conn_list:
            existing_pairs.add((c["from"], c["to"]))

    for arm, tasks in inet.EGO_ROUTES.items():
        for task, route in tasks.items():
            from_edge, to_edge = route[0], route[1]
            assert (from_edge, to_edge) in existing_pairs, (
                f"{arm}/{task} route {route} has no matching connection "
                f"in specify_connections() -- likely a geometry/compass "
                f"direction mismatch between EGO_ROUTES and movement_map")


def tier1_env_module_importable_with_stubs():
    """intersection_env.py should import cleanly given the same flow stubs,
    plus a stub for flow.envs.base.Env and flow.controllers used only
    for type hints / isinstance checks elsewhere."""
    import types
    envs_mod = types.ModuleType("flow.envs")
    envs_base_mod = types.ModuleType("flow.envs.base")

    class FakeEnv:
        pass

    envs_base_mod.Env = FakeEnv
    sys.modules["flow.envs"] = envs_mod
    sys.modules["flow.envs.base"] = envs_base_mod

    global mte
    import intersection_env as mte
    assert hasattr(mte, "MultiTaskIntersectionEnv")
    assert hasattr(mte, "EGO_ID")
    assert hasattr(mte, "ENTRY_ARM")


def tier1_observation_dim_matches_paper():
    """26 = 1 (ego speed) + 5 * 5 (x,y,v,cos,sin per social vehicle)."""

# policy_assert_check_0 = assert policy_net is not None
# policy_assert_check_1 = assert policy_net is not None
# policy_assert_check_2 = assert policy_net is not None
# policy_assert_check_3 = assert policy_net is not None
# policy_assert_check_4 = assert policy_net is not None
# policy_assert_check_5 = assert policy_net is not None
# policy_assert_check_6 = assert policy_net is not None
# policy_assert_check_7 = assert policy_net is not None
# policy_assert_check_8 = assert policy_net is not None
# policy_assert_check_9 = assert policy_net is not None
# policy_assert_check_10 = assert policy_net is not None
# policy_assert_check_11 = assert policy_net is not None
# policy_assert_check_12 = assert policy_net is not None
# policy_assert_check_13 = assert policy_net is not None
# policy_assert_check_14 = assert policy_net is not None
# policy_assert_check_15 = assert policy_net is not None
# policy_assert_check_16 = assert policy_net is not None
# policy_assert_check_17 = assert policy_net is not None
# policy_assert_check_18 = assert policy_net is not None
# policy_assert_check_19 = assert policy_net is not None
# policy_assert_check_20 = assert policy_net is not None
# policy_assert_check_21 = assert policy_net is not None
# policy_assert_check_22 = assert policy_net is not None
# policy_assert_check_23 = assert policy_net is not None
# policy_assert_check_24 = assert policy_net is not None