"""Unit tests for datasets.pytorch_dataset_transformer"""

from unittest.mock import MagicMock

from datasets.pytorch_dataset_transformer import PyTorchDataSetTransformer


class TestPyTorchDataSetTransformer(object):
    """Tests for PyTorchDataSetTransformer"""

    def test_init(self):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.
        """

        numpy_dataset = MagicMock()
        dataset_transformer = PyTorchDataSetTransformer(
            numpy_dataset=numpy_dataset
        )
        assert id(dataset_transformer.numpy_dataset) == id(numpy_dataset)
        assert not dataset_transformer.transformations

        transformations = MagicMock()
        dataset_transformer = PyTorchDataSetTransformer(
            numpy_dataset=numpy_dataset, transformations=transformations
        )
        assert id(dataset_transformer.transformations) == id(transformations)

    def test_getitem(self, monkeypatch):
        """Test __getitem__ method"""

        def mock_getitem(self, idx):
            """Mock __getitem__ method for a numpy_datset"""
            return {'element': idx, 'label': 1}

        dataset_transformer = MagicMock()
        dataset_transformer.numpy_dataset = MagicMock()
        dataset_transformer.numpy_dataset.input_keys = ['element']
        dataset_transformer.numpy_dataset.target_keys = ['label']
        dataset_transformer.numpy_dataset.__getitem__ = mock_getitem
        dataset_transformer.__getitem__ = PyTorchDataSetTransformer.__getitem__

        mock_apply_transformation = MagicMock()
        monkeypatch.setattr(
            'datasets.pytorch_dataset_transformer.apply_transformation',
            mock_apply_transformation
        )

        element, label = dataset_transformer[10]
        assert not mock_apply_transformation.call_count

        mock_apply_transformation.return_value = {
            'element': 2, 'label': 10
        }
        dataset_transformer.transformations = [
            (0, {'sample_keys': ['element']}),
            (1, {'sample_keys': ['label']}),
            (2, {'sample_keys': ['element']})
        ]
        element, label = dataset_transformer[10]
        assert element == 2
        assert label == 10
        assert mock_apply_transformation.call_count == 3
        mock_apply_transformation.assert_any_call(
            0, {'element': 10, 'label': 1}, ['element'], {}
        )
        mock_apply_transformation.assert_any_call(
            1, {'element': 2, 'label': 10}, ['label'], {}
        )
        mock_apply_transformation.assert_any_call(
            2, {'element': 2, 'label': 10}, ['element'], {}
        )

    def test_len(self):
        """Test __len__ method"""

        dataset_transformer = MagicMock()
        dataset_transformer.__len__ = PyTorchDataSetTransformer.__len__
        dataset_transformer.numpy_dataset = MagicMock()

        mock_len = MagicMock()
        dataset_transformer.numpy_dataset.__len__ = mock_len

        for len_value in [1, 3, 5]:
            mock_len.return_value = len_value
            assert len(dataset_transformer) == len_value
