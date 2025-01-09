# %%
import subprocess
from itertools import product
import numpy as np
from sampler import generate


for bike, ebike, pedelec in [(x.item(),y.item(),(1-x-y).item()) for x,y in product(np.arange(0,1.1,0.1), repeat=2) if 0 <= 1-x-y <= 1]:
    scenario = f"{int(100*bike)}_{int(100*ebike)}_{int(100*pedelec)}"
    print(f"running: {bike=}, {ebike=}, {pedelec=}")
    
    generate(scenario, dict(pedestrian=0.0, bike=bike, ebike=ebike, speed_pedelec=pedelec))
    subprocess.run(["cargo", "run", "--", "--scenario", scenario, "--headless"])

