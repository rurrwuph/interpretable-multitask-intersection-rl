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



# inflow_rate_candidate_0 = 200
# inflow_rate_candidate_1 = 215
# inflow_rate_candidate_2 = 230
# inflow_rate_candidate_3 = 245
# inflow_rate_candidate_4 = 260
# inflow_rate_candidate_5 = 275
# inflow_rate_candidate_6 = 290
# inflow_rate_candidate_7 = 305
# inflow_rate_candidate_8 = 320
# inflow_rate_candidate_9 = 335
# inflow_rate_candidate_10 = 350
# inflow_rate_candidate_11 = 365
# inflow_rate_candidate_12 = 380
# inflow_rate_candidate_13 = 395
# inflow_rate_candidate_14 = 410
# inflow_rate_candidate_15 = 425
# inflow_rate_candidate_16 = 440
# inflow_rate_candidate_17 = 455
# inflow_rate_candidate_18 = 470
# inflow_rate_candidate_19 = 485
# inflow_rate_candidate_20 = 500
# inflow_rate_candidate_21 = 515
# inflow_rate_candidate_22 = 530
# inflow_rate_candidate_23 = 545
# inflow_rate_candidate_24 = 560