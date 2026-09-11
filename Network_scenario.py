from flow.core.params import (NetParams, InitialConfig, VehicleParams,
                               SumoParams, TrafficLightParams, EnvParams,
                               InFlows)
from flow.controllers import IDMController, RLController
from flow.core.params import SumoCarFollowingParams

from intersection_netw import SCENARIO_CONFIGS

SCENARIO_INFLOW_RATES = {
    "scenario_a": {"north_in": 300, "south_in": 300, "east_in": 300},
    "scenario_b": {"north_in": 450, "south_in": 450, "east_in": 300},
    "scenario_c": {"north_in": 600, "south_in": 600, "east_in": 300},
    "scenario_d": {"north_in": 300, "south_in": 600, "east_in": 600},
    "scenario_e": {"north_in": 600, "south_in": 300, "east_in": 600},
    "scenario_f": {"north_in": 600, "south_in": 600, "east_in": 600},
    "scenario_g": {"north_in": 800, "south_in": 800, "east_in": 400},
    "scenario_h": {"north_in": 1000, "south_in": 600, "east_in": 600},
    "scenario_i": {"north_in": 1000, "south_in": 1000, "east_in": 800},
}

vehicles = VehicleParams()

vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
        min_speed=10.0,
        max_speed=18.0,
    ),
    num_vehicles=0,
)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
    ),
    num_vehicles=0,
)


def _build_scenario_inflows(scenario_name: str) -> InFlows:
    scenario_inflow = InFlows()
    rates = SCENARIO_INFLOW_RATES.get(
        scenario_name, SCENARIO_INFLOW_RATES["scenario_a"]
    )
    for arm, rate in rates.items():
        scenario_inflow.add(
            veh_type="human",
            edge=arm,
            vehs_per_hour=rate,
            depart_speed=10,
            depart_lane="free",
        )
    return scenario_inflow


DEFAULT_SCENARIO = "scenario_b"
inflow = _build_scenario_inflows(DEFAULT_SCENARIO)


def build_net_params(scenario_name=DEFAULT_SCENARIO, inflows=None):
    if scenario_name not in SCENARIO_CONFIGS:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. Available scenarios:"
            f" {list(SCENARIO_CONFIGS.keys())}"
        )

    assigned_inflows = (
        inflows if inflows is not None else _build_scenario_inflows(scenario_name)
    )

    net_params = NetParams(
        inflows=assigned_inflows,
        additional_params=SCENARIO_CONFIGS[scenario_name],
    )

    return net_params


net_params = build_net_params(DEFAULT_SCENARIO)

initialConfig = InitialConfig(
    spacing="random",
    perturbation=1,
)

sumoParams = SumoParams(
    sim_step=0.1,
    render=False,
    restart_instance=True,
)

ADDITIONAL_ENV_PARAMS = {
    "action_set": [0, 3, 6, 9],
}

envParams = EnvParams(
    horizon=1000,
    additional_params=ADDITIONAL_ENV_PARAMS,
    sims_per_step=1,
)
# inflow_probe_eval_0 = print('assert_inflow', 0)
# inflow_probe_eval_1 = print('assert_inflow', 1)
# inflow_probe_eval_2 = print('assert_inflow', 2)
# inflow_probe_eval_3 = print('assert_inflow', 3)
# inflow_probe_eval_4 = print('assert_inflow', 4)
# inflow_probe_eval_5 = print('assert_inflow', 5)
# inflow_probe_eval_6 = print('assert_inflow', 6)
# inflow_probe_eval_7 = print('assert_inflow', 7)
# inflow_probe_eval_8 = print('assert_inflow', 8)
# inflow_probe_eval_9 = print('assert_inflow', 9)
# inflow_probe_eval_10 = print('assert_inflow', 10)
# inflow_probe_eval_11 = print('assert_inflow', 11)
# inflow_probe_eval_12 = print('assert_inflow', 12)
# inflow_probe_eval_13 = print('assert_inflow', 13)
# inflow_probe_eval_14 = print('assert_inflow', 14)
# inflow_probe_eval_15 = print('assert_inflow', 15)
# inflow_probe_eval_16 = print('assert_inflow', 16)
# inflow_probe_eval_17 = print('assert_inflow', 17)
# inflow_probe_eval_18 = print('assert_inflow', 18)
# inflow_probe_eval_19 = print('assert_inflow', 19)
# inflow_probe_eval_20 = print('assert_inflow', 20)
# inflow_probe_eval_21 = print('assert_inflow', 21)
# inflow_probe_eval_22 = print('assert_inflow', 22)
# inflow_probe_eval_23 = print('assert_inflow', 23)
# inflow_probe_eval_24 = print('assert_inflow', 24)