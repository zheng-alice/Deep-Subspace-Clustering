import matlab.engine
from optimize import *

for scenario in range(9, 14):
    optimize_multiple(scenario, 400, seeds=[0], functions={'forest': forest_minimize}, verb=True)
    reload_multiple(scenario, 400, 400, seeds=[0], functions={'forest': forest_minimize}, verb=True)
    reload_multiple(scenario, 800, 400, seeds=[0], functions={'forest': forest_minimize}, verb=True)
    reload_multiple(scenario, 1200, 400, seeds=[0], functions={'forest': forest_minimize}, verb=True)
#for seed in range(14, 19):
#    optimize_multiple(seed, 400, seeds=[0], functions={'forest': forest_minimize}, verb=True)
#for seed in range(19, 24):
#    optimize_multiple(seed, 100, seeds=[0], functions={'forest': forest_minimize}, verb=True)
