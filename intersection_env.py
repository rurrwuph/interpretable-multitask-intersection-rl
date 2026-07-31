import gymnasium as gym
from gymnasium import spaces
import numpy as np
from flow.envs.base import Env

from intersection_netw import EGO_ROUTES, get_valid_spawn_lane

EGO_ID = "rl_0"
ENTRY_ARM = "west_in"
SAFE_DISTANCE = -100.0  # Safe default for absent social vehicle slots
DISTANCE_BEFORE_STOPLINE = 20.0  # Distance in meters before the junction box


class MultiTaskIntersectionEnv(gym.Env):
    """Gymnasium wrapper around a Flow/SUMO env for multi-task
    unsignalized-intersection navigation (straight/left/right), per
    Kai et al. IV 2020.
    """

    def __init__(self, flow_env: Env):
        super().__init__()
        self.flow_env = flow_env
        self.k_nearest = 5
        self.speed_map = {0: 0.0, 1: 3.0, 2: 6.0, 3: 9.0}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + self.k_nearest * 5,),  # 1 ego + 5*5 social = 26
            dtype=np.float32,
        )

        self.active_g = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.current_task_name = "left"
        self._current_ego_id = None
        self._episode_count = 0
        self._step_in_ep = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_in_ep = 0

        task_choices = {
            "left": np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "straight": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "right": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        }

        if options is not None and "task" in options and options["task"] in task_choices:
            self.current_task_name = options["task"]
            self.active_g = task_choices[self.current_task_name]
        else:
            task_names = list(task_choices.keys())
            idx = np.random.choice(len(task_names))
            self.current_task_name = task_names[idx]
            self.active_g = task_choices[self.current_task_name]

        prev_ego_id = self._current_ego_id
        if prev_ego_id is not None and prev_ego_id in self.flow_env.k.vehicle.get_ids():
            self.flow_env.k.vehicle.remove(prev_ego_id)

        self.flow_env.reset()

        WARMUP_STEPS = 30
        TARGET_ACTIVE_VEHICLES = 30
        for _ in range(WARMUP_STEPS):
            self.flow_env.step(rl_actions=None)
            if len(self.flow_env.k.vehicle.get_ids()) >= TARGET_ACTIVE_VEHICLES:
                break

        ego_route = EGO_ROUTES[ENTRY_ARM][self.current_task_name]

        if prev_ego_id is not None and prev_ego_id in self.flow_env.k.vehicle.get_ids():
            self.flow_env.k.vehicle.remove(prev_ego_id)

        self._episode_count += 1
        self._current_ego_id = f"{EGO_ID}_{self._episode_count}"
        route_name = f"route{self._current_ego_id}_0"
        self.flow_env.k.vehicle.kernel_api.route.add(route_name, ego_route)
        self.flow_env.k.network.rts[self._current_ego_id] = [(ego_route, 1)]

        arm_name = ENTRY_ARM.replace("_in", "")
        arm_cfg = self.flow_env.net_params.additional_params["arm_config"][arm_name]
        spawn_lane = get_valid_spawn_lane(arm_cfg, self.current_task_name)

        # -------------------------------------------------------------
        # Option 1: Calculate Stop-Line Spawn Position & Clear Buffer
        # -------------------------------------------------------------
        try:
            edge_length = self.flow_env.k.network.edge_length(ENTRY_ARM)
        except Exception:
            edge_length = 200.0  # Fallback default

        spawn_pos = max(0.0, edge_length - DISTANCE_BEFORE_STOPLINE)

        # Clear any background vehicle occupying the ego's immediate spawn window
        for veh_id in list(self.flow_env.k.vehicle.get_ids()):
            try:
                if (
                    self.flow_env.k.vehicle.get_edge(veh_id) == ENTRY_ARM
                    and self.flow_env.k.vehicle.get_lane(veh_id) == spawn_lane
                ):
                    pos = self.flow_env.k.vehicle.get_position(veh_id)[0]
                    if (spawn_pos - 10.0) <= pos <= (spawn_pos + 15.0):
                        self.flow_env.k.vehicle.remove(veh_id)
            except Exception:
                pass

    # env_step_probe_0 = print('step', 0)
    # env_step_probe_1 = print('step', 1)
    # env_step_probe_2 = print('step', 2)
    # env_step_probe_3 = print('step', 3)
    # env_step_probe_4 = print('step', 4)
    # env_step_probe_5 = print('step', 5)
    # env_step_probe_6 = print('step', 6)
    # env_step_probe_7 = print('step', 7)
    # env_step_probe_8 = print('step', 8)
    # env_step_probe_9 = print('step', 9)
    # env_step_probe_10 = print('step', 10)
    # env_step_probe_11 = print('step', 11)
    # env_step_probe_12 = print('step', 12)
    # env_step_probe_13 = print('step', 13)
    # env_step_probe_14 = print('step', 14)
    # env_step_probe_15 = print('step', 15)
    # env_step_probe_16 = print('step', 16)
    # env_step_probe_17 = print('step', 17)
    # env_step_probe_18 = print('step', 18)
    # env_step_probe_19 = print('step', 19)
    # env_step_probe_20 = print('step', 20)
    # env_step_probe_21 = print('step', 21)
    # env_step_probe_22 = print('step', 22)
    # env_step_probe_23 = print('step', 23)
    # env_step_probe_24 = print('step', 24)
    # env_step_probe_25 = print('step', 25)
    # env_step_probe_26 = print('step', 26)
    # env_step_probe_27 = print('step', 27)
    # env_step_probe_28 = print('step', 28)
    # env_step_probe_29 = print('step', 29)