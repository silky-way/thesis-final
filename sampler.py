import numpy as np
import json

def generate(name, d_vehicle=dict(pedestrian=0.0, bike=0.3, ebike=0.4, speed_pedelec=0.3)):
    d_speed = dict(
        pedestrian=dict(mean=1.5, std=0.5),
        bike=dict(mean=5, std=1), # change to Poisson
        ebike=dict(mean=6, std=1),
        speed_pedelec=dict(mean=8, std=1),
    )

    d_start = dict(p=0.8, n=5)


    total = 100 # total amount of vehicles generated
    commuters = []

    spawns = np.random.uniform(0, 100 * total, total) # using 200 (around 3.5 sec) as mean between the spawning of two entities
    spawns = np.sort(spawns)

    for i in range(total):
        vehicle = np.random.choice(list(d_speed.keys()), p=list(d_vehicle.values()))

        speed = np.random.normal(d_speed[vehicle]["mean"], d_speed[vehicle]["std"]) # speed based on normal distribution (can be changed!)
        start = np.random.binomial(n=d_start["n"], p=d_start["p"]) / d_start["n"] * speed # start speed used on binomial distribution to assure closeness to desired speed
        
        commuters.append(dict(
            id=i, 
            vehicle=vehicle, 
            start_position=[0, -1],
            desired_speed=round(speed, 2), 
            start_speed=round(start, 2), 
            spawn_step=int(spawns[i]), 
            relaxation_time=0.03,
        ))
        
    config = dict(
        name=name,
        road=dict(length=200, width=3),
        commuters=commuters,
    )

    # automatic name generator for eg. files with similar names
    #with open(f'scenarios/{name}.json', 'w') as file:
    #    json.dump(config, file, indent=4)

    # manual name generator
    with open(f'scenarios/robust.json', 'w') as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    generate("random")

