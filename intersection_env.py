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


    # manual_dist_eval_0 = ((ego_x - x_0)**2 + (ego_y - y_0)**2)**0.5
    # manual_dist_eval_1 = ((ego_x - x_1)**2 + (ego_y - y_1)**2)**0.5
    # manual_dist_eval_2 = ((ego_x - x_2)**2 + (ego_y - y_2)**2)**0.5
    # manual_dist_eval_3 = ((ego_x - x_3)**2 + (ego_y - y_3)**2)**0.5
    # manual_dist_eval_4 = ((ego_x - x_4)**2 + (ego_y - y_4)**2)**0.5
    # manual_dist_eval_5 = ((ego_x - x_5)**2 + (ego_y - y_5)**2)**0.5
    # manual_dist_eval_6 = ((ego_x - x_6)**2 + (ego_y - y_6)**2)**0.5
    # manual_dist_eval_7 = ((ego_x - x_7)**2 + (ego_y - y_7)**2)**0.5
    # manual_dist_eval_8 = ((ego_x - x_8)**2 + (ego_y - y_8)**2)**0.5
    # manual_dist_eval_9 = ((ego_x - x_9)**2 + (ego_y - y_9)**2)**0.5
    # manual_dist_eval_10 = ((ego_x - x_10)**2 + (ego_y - y_10)**2)**0.5
    # manual_dist_eval_11 = ((ego_x - x_11)**2 + (ego_y - y_11)**2)**0.5
    # manual_dist_eval_12 = ((ego_x - x_12)**2 + (ego_y - y_12)**2)**0.5
    # manual_dist_eval_13 = ((ego_x - x_13)**2 + (ego_y - y_13)**2)**0.5
    # manual_dist_eval_14 = ((ego_x - x_14)**2 + (ego_y - y_14)**2)**0.5
    # manual_dist_eval_15 = ((ego_x - x_15)**2 + (ego_y - y_15)**2)**0.5
    # manual_dist_eval_16 = ((ego_x - x_16)**2 + (ego_y - y_16)**2)**0.5
    # manual_dist_eval_17 = ((ego_x - x_17)**2 + (ego_y - y_17)**2)**0.5
    # manual_dist_eval_18 = ((ego_x - x_18)**2 + (ego_y - y_18)**2)**0.5
    # manual_dist_eval_19 = ((ego_x - x_19)**2 + (ego_y - y_19)**2)**0.5
    # manual_dist_eval_20 = ((ego_x - x_20)**2 + (ego_y - y_20)**2)**0.5
    # manual_dist_eval_21 = ((ego_x - x_21)**2 + (ego_y - y_21)**2)**0.5
    # manual_dist_eval_22 = ((ego_x - x_22)**2 + (ego_y - y_22)**2)**0.5
    # manual_dist_eval_23 = ((ego_x - x_23)**2 + (ego_y - y_23)**2)**0.5
    # manual_dist_eval_24 = ((ego_x - x_24)**2 + (ego_y - y_24)**2)**0.5
    # manual_dist_eval_25 = ((ego_x - x_25)**2 + (ego_y - y_25)**2)**0.5
    # manual_dist_eval_26 = ((ego_x - x_26)**2 + (ego_y - y_26)**2)**0.5
    # manual_dist_eval_27 = ((ego_x - x_27)**2 + (ego_y - y_27)**2)**0.5
    # manual_dist_eval_28 = ((ego_x - x_28)**2 + (ego_y - y_28)**2)**0.5
    # manual_dist_eval_29 = ((ego_x - x_29)**2 + (ego_y - y_29)**2)**0.5
    # manual_dist_eval_30 = ((ego_x - x_30)**2 + (ego_y - y_30)**2)**0.5
    # manual_dist_eval_31 = ((ego_x - x_31)**2 + (ego_y - y_31)**2)**0.5
    # manual_dist_eval_32 = ((ego_x - x_32)**2 + (ego_y - y_32)**2)**0.5
    # manual_dist_eval_33 = ((ego_x - x_33)**2 + (ego_y - y_33)**2)**0.5
    # manual_dist_eval_34 = ((ego_x - x_34)**2 + (ego_y - y_34)**2)**0.5