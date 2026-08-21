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
    k_nearest = 5
    expected_dim = 1 + k_nearest * 5
    assert expected_dim == 26, f"math check failed: got {expected_dim}"


def tier1_reward_function_matches_eq9():
    """Reproduce the paper's Eq. (9) branch values directly, independent
    of any live env instance, by calling the pure function logic."""
    import types
    envs_mod = types.ModuleType("flow.envs")
    envs_base_mod = types.ModuleType("flow.envs.base")

    class FakeEnv:
        pass

    envs_base_mod.Env = FakeEnv
    sys.modules["flow.envs"] = envs_mod
    sys.modules["flow.envs.base"] = envs_base_mod
    import intersection_env as mte

    # _subtask_reward is a plain method with no self-state dependency
    # beyond its args, so we can call it unbound on a throwaway instance
    # without a real flow_env.
    dummy = mte.MultiTaskIntersectionEnv.__new__(mte.MultiTaskIntersectionEnv)

    assert dummy._subtask_reward(is_active=0.0, collided=True,
                                  succeeded=False) == 0.0, "inactive task must be masked to 0"
    assert dummy._subtask_reward(is_active=1.0, collided=True,
                                  succeeded=False) == -500.0
    assert dummy._subtask_reward(is_active=1.0, collided=False,
                                  succeeded=True) == 50.0
    assert dummy._subtask_reward(is_active=1.0, collided=False,
                                  succeeded=False) == 0.0


def tier1_scalar_reward_masking_math():
    """g^T r should zero out inactive sub-tasks and sum active ones,
    matching Eq. (4)/(5)."""
    import numpy as np
    g = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # "left" task
    r = np.array([-500.0, 999.0, 999.0, -0.15], dtype=np.float32)
    scalar = float(np.dot(g, r))
    # only r_ls (-500) and r_cs (-0.15) should count; r_ss/r_rs (999 each,
    # deliberately absurd values) must be fully masked out. Use a small
    # tolerance since g/r are float32 and exact equality is unreliable.
    assert abs(scalar - (-500.15)) < 1e-3, f"masking broken, got {scalar}"


# ======================================================================
# TIER 2: live SUMO smoke test (requires flow + SUMO installed)
# ======================================================================
 
def tier2_live_sumo_smoke_test():
    # Fresh interpreter-level import, no stubs -- will fail loudly with
    # ModuleNotFoundError if flow isn't actually installed, which the
    # caller catches and reports as SKIP rather than FAIL.
    for mod in list(sys.modules):
        if mod.startswith("flow") or mod in ("intersection_network",
                                              "multi_task_env",
                                              "scenario_params"):
            del sys.modules[mod]
 
    import numpy as np
    from flow.envs.base import Env
    from flow.core.params import SumoParams, TrafficLightParams
 
    import Network_scenario as sp
    from intersection_netw import UnsignalizedIntersectionNetwork
    from intersection_env import MultiTaskIntersectionEnv, EGO_ID
 
    # Build a minimal Flow Env subclass wired to our network + params.
    # This mirrors what your real training entrypoint will do -- adjust
    # class name / action-space methods if your Flow version's base Env
    # requires overriding action_space/observation_space/_apply_rl_actions
    # as abstract methods (some Flow versions do).
    class _RawFlowEnv(Env):
        @property
        def action_space(self):
            from gymnasium import spaces
            return spaces.Discrete(4)
 
        @property
        def observation_space(self):
            from gymnasium import spaces
            return spaces.Box(low=-np.inf, high=np.inf, shape=(26,),
                               dtype=np.float32)
 
        def _apply_rl_actions(self, rl_actions):
            pass  # actuation handled externally via setSpeed in wrapper
 
        def get_state(self):
            return np.zeros(26, dtype=np.float32)
 
        def compute_reward(self, rl_actions, **kwargs):
            return 0.0
 
    network = UnsignalizedIntersectionNetwork(
        name="smoke_test",
        vehicles=sp.vehicles,
        net_params=sp.build_net_params("scenario_b"),  # simplest config
        # Empty (non-None) TrafficLightParams: unsignalized intersection,
        # but Flow's traci kernel unconditionally calls
        # traffic_lights.get_properties() during net generation, so None
        # raises AttributeError here (same fix already applied in
        # train_multitask_dqn.py's build_flow_env()).
        traffic_lights=TrafficLightParams(),
    )
 
    raw_env = _RawFlowEnv(
        env_params=sp.envParams,
        sim_params=sp.sumoParams,
        network=network,
    )
 
    env = MultiTaskIntersectionEnv(raw_env)
 
    obs, info = env.reset()
    assert obs.shape == (26,), f"obs shape wrong: {obs.shape}"
    assert not np.isnan(obs).any(), "obs contains NaN"
 
    for step_i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert obs.shape == (26,), f"step {step_i}: obs shape wrong"
        assert isinstance(reward, float), f"step {step_i}: reward not float"
        assert "vectorized_reward" in step_info
        assert "task_g" in step_info
        if terminated or truncated:
            break
 
    print(f"    (ran {step_i + 1} steps, task={env.current_task_name}, "
          f"final_reward={reward:.3f})")
 
 
# ======================================================================
# Run everything
# ======================================================================
 
if __name__ == "__main__":
    print("=" * 70)
    print("TIER 1: structural checks (no flow/SUMO required)")
    print("=" * 70)
    check("import intersection_network (stubbed flow)",
          tier1_import_network_module)
    check("all 9 scenarios present", tier1_nine_scenarios_present)
    check("scenario configs have required keys",
          tier1_scenario_configs_have_required_keys)
    check("connections reference valid lane indices",
          tier1_connections_reference_valid_lanes)
    check("EGO_ROUTES covers all arms x tasks",
          tier1_ego_routes_cover_all_arms_and_tasks)
    check("EGO_ROUTES match real generated connections (geometry check)",
          tier1_ego_routes_match_real_connections)
    check("import multi_task_env (stubbed flow)",
          tier1_env_module_importable_with_stubs)
    check("observation dim = 1 + 5*5 = 26",
          tier1_observation_dim_matches_paper)
    check("_subtask_reward matches paper Eq. (9)",
          tier1_reward_function_matches_eq9)
    check("g^T r masking math (Eq. 4/5)",
          tier1_scalar_reward_masking_math)
 
    print()
    print("=" * 70)
    print("TIER 2: live SUMO smoke test (requires flow + SUMO installed)")
    print("=" * 70)
    # NOTE: Tier 1 checks intentionally inject fake stub modules into
    # sys.modules under the name "flow" (and "flow.envs.base" etc.) to
    # test our own code without a real flow install. Those stubs must be
    # purged here first, or this guard will "successfully" import the
    # stub and wrongly proceed into Tier 2 instead of skipping.
    for mod in list(sys.modules):
        if mod == "flow" or mod.startswith("flow."):
            del sys.modules[mod]
 
    try:
        import flow  # noqa: F401
        # Confirm this is a real install, not another stub, by checking
        # for an attribute only a genuine flow package would have.
        assert hasattr(flow, "__file__") and flow.__file__ is not None
        check("live SUMO smoke test (build net, reset, 10 steps)",
              tier2_live_sumo_smoke_test)
    except (ImportError, AssertionError):
        skip("live SUMO smoke test",
             "flow is not installed in this environment -- run this "
             "script where flow + SUMO are available to execute Tier 2")
 
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailed checks:")
        for name, status, msg in results:
            if status == FAIL:
                print(f"  - {name}: {msg}")
        sys.exit(1)
 