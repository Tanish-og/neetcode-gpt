import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        j = 0
        n = len(y_true)
        ans = 0

        for i in range(len(y_true)):
            if y_true[i] == 1:
                ans = ans + np.log(y_pred[j])
            else:
                ans = ans + np.log(1 - y_pred[j])

            j = j + 1

        ans = ans * (-1 / n)

        return round(ans, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: one-hot encoded true labels
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        n = len(y_true)
        ans = 0

        for i in range(n):
            for j in range(len(y_true[i])):
                if y_true[i][j] == 1:
                    ans += np.log(y_pred[i][j] + 1e-7)

        ans = -ans / n

        return round(ans, 4)