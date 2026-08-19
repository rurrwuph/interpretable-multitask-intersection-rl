"""
Smoke test to verify environment instantiation, reset, and step execution
across all 9 unsignalized intersection scenarios (a through i).
Runs 5 episodes per scenario.
"""

import sys
import numpy as np
import torch
from Network_scenario import (
    vehicles,
    build_net_params,
    initialConfig,
    sumoParams,
    envParams,
    SCENARIO_INFLOW_RATES,
)
from intersection_netw import UnsignalizedIntersectionNetwork, SCENARIO_CONFIGS
from intersection_env import MultiTaskIntersectionEnv
from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams
from gymnasium import spaces

N_ACTIONS = 4
EXPECTED_OBS_DIM = 26
EPISODES_PER_SCENARIO = 5
MAX_TEST_STEPS = 100


def build_test_env(scenario_name: str) -> MultiTaskIntersectionEnv:
    class _RawFlowEnv(FlowEnv):
        @property
        def action_space(self):
            return spaces.Discrete(N_ACTIONS)

        @property
        def observation_space(self):
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(EXPECTED_OBS_DIM,), dtype=np.float32
            )

        def _apply_rl_actions(self, rl_actions):
            pass

        def get_state(self):
            return np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)

        def compute_reward(self, rl_actions, **kwargs):
            return 0.0

    net_params = build_net_params(scenario_name)
    network = UnsignalizedIntersectionNetwork(
        name=f"smoke_{scenario_name}",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initialConfig,
        traffic_lights=TrafficLightParams(),
    )
    raw_env = _RawFlowEnv(
        env_params=envParams,
        sim_params=sumoParams,
        network=network,
    )
    return MultiTaskIntersectionEnv(raw_env)


def run_smoke_test():
    scenarios = [f"scenario_{c}" for c in "abcdefghi"]
    print("=" * 70)
    print(f"STARTING SMOKE TEST: {len(scenarios)} Scenarios | {EPISODES_PER_SCENARIO} Eps Each")
    print("=" * 70)

    results = {}

    for scenario_name in scenarios:
        print(f"\n[SCENARIO] Initializing '{scenario_name}'...")
        rates = SCENARIO_INFLOW_RATES.get(scenario_name, {})
        print(f"           Flow rates (vehs/hr): {rates}")

        try:
            env = build_test_env(scenario_name)
        except Exception as e:
            print(f"[FAIL] Error building environment for {scenario_name}: {e}")
            results[scenario_name] = "BUILD_ERROR"
            continue

        ep_successes = 0
        ep_collisions = 0
        ep_truncations = 0

        for ep in range(1, EPISODES_PER_SCENARIO + 1):
            try:
                obs, _ = env.reset()
            except Exception as e:
                print(f"  [Ep {ep}] Reset failed: {e}")
                results[scenario_name] = "RESET_ERROR"
                break

            # Validate observation vector shape and finite values
            assert obs.shape == (EXPECTED_OBS_DIM,), (
                f"Invalid obs shape {obs.shape}, expected ({EXPECTED_OBS_DIM},)"
            )
            assert not np.isnan(obs).any(), "NaN detected in observation vector"

            task_name = env.current_task_name
            g_vec = env.active_g

            done = False
            step_count = 0

            while not done and step_count < MAX_TEST_STEPS:
                action = np.random.randint(0, N_ACTIONS)
                next_obs, scalar_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                assert next_obs.shape == (EXPECTED_OBS_DIM,), "Invalid next_obs shape"
                assert "vectorized_reward" in info, "Missing vectorized_reward in info"
                assert len(info["vectorized_reward"]) == 4, "Vectorized reward must have length 4"

                obs = next_obs
                step_count += 1

            is_succ = info.get("is_success", False)
            is_coll = info.get("is_collision", False)
            coll_loc = info.get("collision_type", "None")

            if is_succ:
                ep_successes += 1
            if is_coll:
                ep_collisions += 1
            if not is_succ and not is_coll:
                ep_truncations += 1

            status = "SUCCESS" if is_succ else (f"COLLISION ({coll_loc})" if is_coll else "TIMEOUT")
            print(
                f"  [Ep {ep}/{EPISODES_PER_SCENARIO}] Task: {task_name:8s} | "
                f"Steps: {step_count:3d} | Result: {status}"
            )

        results[scenario_name] = (
            f"Pass ({ep_successes}S / {ep_collisions}C / {ep_truncations}T)"
            if scenario_name not in results
            else results[scenario_name]
        )

        # Teardown SUMO instance before next scenario
        try:
            env.flow_env.terminate()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for sc, res in results.items():
        print(f"  - {sc:12s} : {res}")
        if "ERROR" in res:
            all_passed = False

    if all_passed:
        print("\nAll 9 scenarios initialized and executed successfully.")
    else:
        print("\nSmoke test encountered errors. Review the trace above.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()