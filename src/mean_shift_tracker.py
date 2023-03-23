import numpy as np
from src.ex2_utils import Tracker, get_patch, extract_histogram, \
    create_epanechnik_kernel, backproject_histogram
from src.mean_shift import mean_shift

class MeanShiftParams():
    def __init__(self):
        self.kernel_sigma = 0.005
        self.nbins = 16
        self.histogram_update_alpha = 0
        self.ms_epsilon = 0.01

class MeanShiftTracker(Tracker):
    parameters: MeanShiftParams

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_),
                      np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        
        if(self.size[0]%2 == 0):
            self.size = (self.size[0]+1, self.size[1])
        if(self.size[1]%2 == 0):
            self.size = (self.size[0], self.size[1]+1)

        # Create kernel as it will be used always
        self.kernel = create_epanechnik_kernel(self.size[0],self.size[1],
                                               self.parameters.kernel_sigma)

        # Create first histogram (GT)
        patch, mask = get_patch(image, self.position, self.size)
        mask = np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2)
        patch = np.multiply(patch, mask)
        self.histogram = extract_histogram(patch, self.parameters.nbins, self.kernel)

        # Normalize histogram
        self.histogram = self._normalize_histogram(self.histogram)

    def track(self, image):
        # Get patch from previous window
        patch, mask = get_patch(image, self.position, self.size)
        mask = np.repeat(mask.reshape((mask.shape[0], mask.shape[1], 1)), 3, axis=2)
        patch = np.multiply(patch, mask)

        # Extract the histogram of the current patch
        current_histogram = extract_histogram(patch, self.parameters.nbins,
                                              self.kernel)
        
        # Normalize histogram
        current_histogram = self._normalize_histogram(current_histogram)

        # weight calculation
        q = backproject_histogram(image, self.histogram, self.parameters.nbins)
        p = backproject_histogram(image, current_histogram, self.parameters.nbins)
        p += 1e-5

        weights = np.divide(q, p)
        weights = np.sqrt(weights)

        # Run mean shift
        new_position, _, _ = mean_shift(weights, self.size, self.position, 
                                        self.parameters.ms_epsilon, 20)
        self.position = new_position

        # Update histogram
        self.histogram = (1-self.parameters.histogram_update_alpha)*self.histogram + self.parameters.histogram_update_alpha*current_histogram
        self.histogram = self._normalize_histogram(self.histogram)
    
        return [self.position[0], self.position[1], self.size[0], self.size[1]]

    def _normalize_histogram(self, histogram: np.array) -> np.array:
        return histogram / np.linalg.norm(histogram)