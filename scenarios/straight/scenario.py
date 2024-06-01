from pathlib import Path
import os
import numpy as np
from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
)


traffic={}
a=1
# Traffic Flows
for seed in np.random.choice(1000, 20, replace=False):
    actors = {}

    for i in range(4):
        car = TrafficActor(
            name = f'car_type_{i+1}',
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(speed_gain=np.random.uniform(1.0, 2.0), impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
            depart_speed=8,
            max_speed=8,
        )

        actors[car] = 0.25
    actors1 = {}

    for i in range(4):
        car = TrafficActor(
            name = f'car1_type_{i+1}',
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(speed_gain=np.random.uniform(1.0, 2.0), impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
            depart_speed=14,
            max_speed=14,
        )

        actors1[car] = 0.25        

    west_east_flow = [Flow(route=Route(begin=("E0", 0, "random"), end=(f"E0", i, "max")),
                           rate=250, actors=actors) for i in range(3)]
    east_west_flow = [Flow(route=Route(begin=("E0", 1, "random"), end=(f"E0", 1, "max")),
                           rate=150, actors=actors) for i in range(2)]
    turn_right_flow = [Flow(route=Route(begin=("E0", 2, "random"), end=(f"E0", 2, "max")),
                            rate=100, actors=actors1) for i in range(3)]
    a=a+1
    traffic[str(a)] = Traffic(flows = west_east_flow + east_west_flow + turn_right_flow)
    

# Agent Missions
route = Route(begin=("E0", 0, "0"), end=("E0", 1, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=3,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
