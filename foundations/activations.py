import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans = np.empty(len(z))
        for i in range(len(z)):
            ans[i] = 1 / (1 + np.exp(-z[i]))
            ans[i]=np.round(ans[i],5)
        return ans
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans = np.empty(len(z))
        for i in range(len(z)):
            ans[i]=max(0,z[i])
            ans[i]=np.round(ans[i],5)
        
        return ans
        pass
