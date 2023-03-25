from src.run_tracker import run_tracker
from src.mean_shift_tracker import MeanShiftParams
parameters = MeanShiftParams(kernel_sigma=0.05, histogram_update_alpha=0.1, steps=3, nbins=24)
print(run_tracker("./data", "singer", parameters))