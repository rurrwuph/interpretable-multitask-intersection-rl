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

        self.flow_env.k.vehicle.add(
            veh_id=self._current_ego_id,
            type_id="rl",
            edge=ENTRY_ARM,
            lane=spawn_lane,
            pos=spawn_pos,
            speed=0.0,
        )

        self.flow_env.k.vehicle.kernel_api.vehicle.setSpeedMode(self._current_ego_id, 0)

        # Advance 1 simulation step so TraCI registers ego geometry
        self.flow_env.step(rl_actions=None)

        obs = self._get_observation()
        return obs, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int):
        self._step_in_ep += 1
        target_speed = self.speed_map[int(action)]

        ego_present_pre = self._current_ego_id in self.flow_env.k.vehicle.get_ids()
        prev_speed = (
            float(self.flow_env.k.vehicle.get_speed(self._current_ego_id))
            if ego_present_pre else 0.0
        )

        if ego_present_pre:
            self.flow_env.k.vehicle.kernel_api.vehicle.setSpeed(
                self._current_ego_id, target_speed
            )

        self.flow_env.step(rl_actions=None)

        ego_present = self._current_ego_id in self.flow_env.k.vehicle.get_ids()
        current_speed = (
            float(self.flow_env.k.vehicle.get_speed(self._current_ego_id))
            if ego_present else 0.0
        )

        dt = self.flow_env.sim_params.sim_step
        realized_decel = (prev_speed - current_speed) / dt if dt > 0 else 0.0

        # Fetch edge before SUMO cleans up the vehicle
        if ego_present:
            try:
                ego_edge = self.flow_env.k.vehicle.get_edge(self._current_ego_id)
                ego_lane = self.flow_env.k.vehicle.get_lane(self._current_ego_id)
            except Exception:
                ego_edge = None
                ego_lane = None
        else:
            ego_edge = None
            ego_lane = None

        # Hard Braking Telemetry
        if ego_present and realized_decel > 4.5:
            try:
                sim_time = self.flow_env.k.simulation.kernel_api.simulation.getTime()
            except Exception:
                sim_time = self._step_in_ep * dt

            try:
                leader_info = self.flow_env.k.vehicle.kernel_api.vehicle.getLeader(
                    self._current_ego_id, 100.0
                )
                leader_str = (
                    f"Leader: '{leader_info[0]}' (gap={leader_info[1]:.2f}m)"
                    if leader_info else "Leader: None"
                )
            except Exception:
                leader_str = "Leader: <unavailable>"

            print(
                f"[BRAKE TELEMETRY | Ep {self._episode_count} Step {self._step_in_ep}] "
                f"Time: {sim_time:5.2f}s | Lane: {ego_edge}_{ego_lane} | "
                f"Action Chosen: {action} (Target: {target_speed} m/s) | "
                f"Speed: {prev_speed:.2f} -> {current_speed:.2f} m/s | "
                f"Decel: {realized_decel:.2f} m/s² | {leader_str}",
                flush=True
            )

        # Collision Query & Localization
        collided_ids = (
            self.flow_env.k.simulation.kernel_api.simulation
            .getCollidingVehiclesIDList()
        )
        collided = self._current_ego_id in collided_ids

        collision_type = "None"
        if collided:
            if ego_edge is not None and ego_edge.startswith(":"):
                collision_type = "AT_INTERSECTION"
            elif ego_edge == ENTRY_ARM:
                collision_type = "BEFORE_INTERSECTION (Entry Arm)"
            else:
                collision_type = f"OTHER ({ego_edge})"

            print(
                f"[COLLISION EVENT | Ep {self._episode_count} Step {self._step_in_ep}] "
                f"Location: {collision_type} | Edge: {ego_edge}",
                flush=True
            )

        target_exit_edge = EGO_ROUTES[ENTRY_ARM][self.current_task_name][-1]
        succeeded = (ego_edge == target_exit_edge)

        r_ls = self._subtask_reward(self.active_g[0], collided, succeeded)
        r_ss = self._subtask_reward(self.active_g[1], collided, succeeded)
        r_rs = self._subtask_reward(self.active_g[2], collided, succeeded)
        r_cs = -0.15

        vectorized_reward = np.array([r_ls, r_ss, r_rs, r_cs], dtype=np.float32)
        scalar_reward = float(np.dot(self.active_g, vectorized_reward))

        terminated = bool(collided or succeeded)
        truncated = bool(
            self.flow_env.step_counter >= self.flow_env.env_params.horizon
        )

        obs = self._get_observation()
        info = {
            "vectorized_reward": vectorized_reward,
            "task_g": self.active_g,
            "is_success": succeeded,
            "is_collision": collided,
            "collision_type": collision_type,
            "edge": ego_edge,
        }

        return obs, scalar_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _subtask_reward(self, is_active, collided, succeeded):
        if not is_active:
            return 0.0
        if collided:
            return -500.0
        if succeeded:
            return 50.0
        return 0.0

    def _get_observation(self):
        all_veh_ids = self.flow_env.k.vehicle.get_ids()
        ego_present = (self._current_ego_id is not None) and (self._current_ego_id in all_veh_ids)

        ego_pos = np.array([0.0, 0.0], dtype=np.float32)
        ego_speed = 0.0
        ego_angle = 0.0

        if ego_present:
            try:
                raw_pos = self.flow_env.k.vehicle.get_position(self._current_ego_id)
                if raw_pos is not None and len(raw_pos) >= 2:
                    ego_pos = np.array(raw_pos[:2], dtype=np.float32)

                ego_speed = float(self.flow_env.k.vehicle.get_speed(self._current_ego_id))
                ego_angle = float(
                    self.flow_env.k.vehicle.kernel_api.vehicle.getAngle(self._current_ego_id)
                )
            except Exception:
                ego_present = False

        ego_state = [max(0.0, ego_speed)]

        social_ids = [
            v for v in all_veh_ids
            if v.startswith("human") or v.startswith("flow")
        ]
        social_features = []

        if len(social_ids) > 0 and ego_present:
            distances = []
            for s_id in social_ids:
                try:
                    s_raw_pos = self.flow_env.k.vehicle.get_position(s_id)
                    if s_raw_pos is None or len(s_raw_pos) < 2:
                        continue
                    s_pos = np.array(s_raw_pos[:2], dtype=np.float32)
                    dist = float(np.linalg.norm(s_pos - ego_pos))
                    distances.append((dist, s_id, s_pos))
                except Exception:
                    continue

            distances.sort(key=lambda x: x[0])
            nearest = distances[: self.k_nearest]

            ego_rad = np.radians(ego_angle)

            for dist, s_id, s_pos in nearest:
                try:
                    s_speed = float(self.flow_env.k.vehicle.get_speed(s_id))
                    s_angle = float(
                        self.flow_env.k.vehicle.kernel_api.vehicle.getAngle(s_id)
                    )
                except Exception:
                    s_speed = 0.0
                    s_angle = ego_angle

                dx = float(s_pos[0] - ego_pos[0])
                dy = float(s_pos[1] - ego_pos[1])

                x_body = dx * np.sin(ego_rad) + dy * np.cos(ego_rad)
                y_body = -dx * np.cos(ego_rad) + dy * np.sin(ego_rad)

                rel_angle = s_angle - ego_angle
                cos_i = float(np.cos(np.radians(rel_angle)))
                sin_i = float(np.sin(np.radians(rel_angle)))

                social_features.extend([x_body, y_body, s_speed, cos_i, sin_i])

        while len(social_features) < self.k_nearest * 5:
            social_features.extend([SAFE_DISTANCE, 0.0, 0.0, 1.0, 0.0])

        obs_vector = np.array(ego_state + social_features, dtype=np.float32)
        return obs_vector
# catch sumo teleportation and collision events
