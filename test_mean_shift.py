import numpy as np

from src.ex2_utils import generate_responses_1, get_patch
from src.mean_shift import mean_shift, visualize_mean_shift


density = generate_responses_1()
# print(density[71, 51])
# print(get_patch(density, (70, 50), (5,5)))
pos, iter, positions = mean_shift(density, (5, 5), (55, 75))
print(positions)

visualize_mean_shift(density, positions)
