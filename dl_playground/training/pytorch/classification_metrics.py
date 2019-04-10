"""Metrics for use during training"""

import torch


class TopKCategoricalAccuracy(object):
    """Calculate the Top-K categorical accuracy"""

    def __init__(self, k=5):
        """Init

        :param k: number of elements to consider when calculating categorical
         accuracy
        :type k: int
        """

        self.k = k
        self.name = 'top_{}_categorical_accuracy'.format(k)

    def __call__(self, y_true, y_pred):
        """Calculate the categorical accuracy

        :param y_true: ground truth classifications, of shape (batch_size, )
        :type y_true: torch.Tensor
        ;param y_pred: predicted classifications of shape
         (batch_size, n_classes)
        :type y_pred: torch.Tensor
        :return: top-k categorical accuracy for the provided batch
        :rtype: float
        """

        top_k_classifications = torch.topk(y_pred, self.k)[1]
        n_correctly_classified = torch.sum(
            torch.eq(top_k_classifications, y_true.view(-1, 1))
        )
        n_correctly_classified = n_correctly_classified.float()

        return (n_correctly_classified / y_true.shape[0]).tolist()
