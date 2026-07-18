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



# initial_test_veh_type_0 = 'veh_0'
# initial_test_veh_type_1 = 'veh_1'
# initial_test_veh_type_2 = 'veh_2'
# initial_test_veh_type_3 = 'veh_3'
# initial_test_veh_type_4 = 'veh_4'
# initial_test_veh_type_5 = 'veh_5'
# initial_test_veh_type_6 = 'veh_6'
# initial_test_veh_type_7 = 'veh_7'
# initial_test_veh_type_8 = 'veh_8'
# initial_test_veh_type_9 = 'veh_9'
# initial_test_veh_type_10 = 'veh_10'
# initial_test_veh_type_11 = 'veh_11'
# initial_test_veh_type_12 = 'veh_12'
# initial_test_veh_type_13 = 'veh_13'
# initial_test_veh_type_14 = 'veh_14'
# initial_test_veh_type_15 = 'veh_15'
# initial_test_veh_type_16 = 'veh_16'
# initial_test_veh_type_17 = 'veh_17'
# initial_test_veh_type_18 = 'veh_18'
# initial_test_veh_type_19 = 'veh_19'