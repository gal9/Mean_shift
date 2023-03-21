import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.ex2_utils import get_patch

def x_i_window_construction(window_size: Tuple[int, int]) -> Tuple[np.array, np.array]:
    # Check that the window size is odd
    if(window_size[0]%2 == 0 or window_size[1]%2 == 0):
        print("Window size must bi an odd number.")
        return None
    
    # Calculate half window size
    x_half = (window_size[0]-1)/2
    y_half = (window_size[1]-1)/2
    
    # Construct a single row or column of the final matrix
    x_i = np.arange(-x_half, x_half+1, 1, dtype=np.float32)
    y_i = np.arange(-y_half, y_half+1, 1, dtype=np.float32)

    # Repeat this column/row to get the matrix
    x_i = np.repeat([x_i], window_size[1], axis=0)
    y_i = np.repeat([y_i], window_size[0], axis=0).transpose()

    return x_i, y_i

def mean_shift(density: np.array, kernel_size: Tuple[int, int], start_position: Tuple[int, int]):
    x_i, y_i = x_i_window_construction(kernel_size)

    # PREPARATION FOR ITERATIONS

    # Initialize just so it would go into iteration
    x_change = 1
    y_change = 1

    current_position_x, current_position_y = start_position
    iteration = 0
    positions = []

    # ITERATIONS
    while(abs(x_change) >= 0.01 or abs(y_change) >= 0.01):
        positions.append((current_position_x, current_position_y))

        mask, patch = get_patch(density, (current_position_x, current_position_y), kernel_size)
        patch = np.multiply(mask, patch)

        x_change = np.divide(np.sum(np.multiply(patch, x_i)), np.sum(patch))
        y_change = np.divide(np.sum(np.multiply(patch, y_i)), np.sum(patch))

        current_position_x += round_larger(x_change)
        current_position_y += round_larger(y_change)

        # Increase the number of iterations
        iteration += 1

    return (current_position_x, current_position_y), iteration, positions

def round_larger(n: float) -> int:
    if(n>0):
        return math.ceil(n)
    else:
        return math.floor(n)

def visualize_mean_shift(density: np.array, path: List[Tuple[int, int]]) -> None:
    plt.imshow(density)

    x = [p[0] for p in path]
    y = [p[1] for p in path]

    plt.scatter(x, y, color="red")

    plt.show()