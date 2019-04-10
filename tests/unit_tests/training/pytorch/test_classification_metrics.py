"""Unit tests for training.pytorch.classification_metrics"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from training.pytorch.classification_metrics import TopKCategoricalAccuracy


class TestTopKCategoricalAccuracy(object):
    """Tests for TopKCategoricalAccuracy"""

    @pytest.fixture(scope='class')
    def y_true(self):
        """"y_true object fixture

        Ground truth classifications will be one of 5 classes.

        :return: array of ground truth classifications of shape (10, )
        :rtype: torch.Tensor
        """

        y_true = np.array([1, 3, 4, 2, 0, 1, 3, 2, 0, 4])
        y_true = torch.from_numpy(y_true)

        return y_true

    @pytest.fixture(scope='class')
    def y_pred(self):
        """y_pred object_fixture

        Predicted classifications will be for five separate classes.

        :return: array of ground truth classifications of shape (10, 5)
        :rtype: torch.Tensor
        """

        y_pred = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.2, 0.3, 0.3, 0.075, 0.125],
            [0.025, 0.05, 0.075, 0.15, 0.70],
            [0.2, 0.2, 0.2, 0.3, 0.1],
            [0.2, 0.3, 0.1, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.2, 0.3, 0.3, 0.075, 0.125],
            [0.025, 0.05, 0.075, 0.15, 0.70],
            [0.2, 0.2, 0.2, 0.3, 0.1],
            [0.2, 0.3, 0.1, 0.25, 0.25],
        ])
        y_pred = torch.from_numpy(y_pred)

        return y_pred

    def test_init(self):
        """Test __init__

        This simply tests that all attributes are set correctly.
        """

        test_cases = [
            {'k':  1, 'expected_name': 'top_1_categorical_accuracy'},
            {'k':  2, 'expected_name': 'top_2_categorical_accuracy'},
            {'k':  4, 'expected_name': 'top_4_categorical_accuracy'},
            {'k':  5, 'expected_name': 'top_5_categorical_accuracy'}
        ]

        for test_case in test_cases:
            k = test_case['k']
            metric = TopKCategoricalAccuracy(k=k)

            assert metric.k == k
            assert metric.name == test_case['expected_name']

    def test_call__k_of_1(self, y_true, y_pred):
        """Test __call__ method when k=1

        :param y_true: y_true object fixture
        :type y_true: torch.Tensor
        :param y_pred: y_predobject fixture
        :type y_pred: torch.Tensor
        """

        mock_metric = MagicMock()
        mock_metric.k = 1
        mock_metric.__call__ = TopKCategoricalAccuracy.__call__

        expected_categorical_accuracy = 0.10
        categorical_accuracy = mock_metric.__call__(
            self=mock_metric, y_true=y_true, y_pred=y_pred
        )

        assert np.isclose(categorical_accuracy, expected_categorical_accuracy)

    def test_call__k_of_2(self, y_true, y_pred):
        """Test __call__ method when k=2

        :param y_true: y_true object fixture
        :type y_true: torch.Tensor
        :param y_pred: y_predobject fixture
        :type y_pred: torch.Tensor
        """

        mock_metric = MagicMock()
        mock_metric.k = 2
        mock_metric.__call__ = TopKCategoricalAccuracy.__call__

        expected_categorical_accuracy = 0.30
        categorical_accuracy = mock_metric.__call__(
            self=mock_metric, y_true=y_true, y_pred=y_pred
        )

        assert np.isclose(categorical_accuracy, expected_categorical_accuracy)

    def test_call__k_of_4(self, y_true, y_pred):
        """Test __call__ method when k=4

        :param y_true: y_true object fixture
        :type y_true: torch.Tensor
        :param y_pred: y_predobject fixture
        :type y_pred: torch.Tensor
        """

        mock_metric = MagicMock()
        mock_metric.k = 4
        mock_metric.__call__ = TopKCategoricalAccuracy.__call__

        expected_categorical_accuracy = 0.80
        categorical_accuracy = mock_metric.__call__(
            self=mock_metric, y_true=y_true, y_pred=y_pred
        )

        assert np.isclose(categorical_accuracy, expected_categorical_accuracy)

    def test_call__k_of_5(self, y_true, y_pred):
        """Test __call__ method when k=5

        :param y_true: y_true object fixture
        :type y_true: torch.Tensor
        :param y_pred: y_predobject fixture
        :type y_pred: torch.Tensor
        """

        mock_metric = MagicMock()
        mock_metric.k = 5
        mock_metric.__call__ = TopKCategoricalAccuracy.__call__

        expected_categorical_accuracy = 1.0
        categorical_accuracy = mock_metric.__call__(
            self=mock_metric, y_true=y_true, y_pred=y_pred
        )

        assert np.isclose(categorical_accuracy, expected_categorical_accuracy)
