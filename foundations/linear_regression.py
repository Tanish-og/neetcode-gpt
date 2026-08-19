import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        predictions = X @ weights

        return np.round(predictions, 5)
        

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        ans=0
        n=len(model_prediction)
        for i in range(n):
            ans=ans+(model_prediction[i][0]-ground_truth[i][0])*(model_prediction[i][0]-ground_truth[i][0])
            
        ans=ans/n
        return round(float(ans),5)

        
