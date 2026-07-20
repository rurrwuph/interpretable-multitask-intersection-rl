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



# ego vehicle rlcontroller configured
