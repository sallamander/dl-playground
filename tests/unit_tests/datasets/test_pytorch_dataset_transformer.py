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
        dataset_transformer.numpy_dataset.__getitem__ = mock_getitem
        dataset_transformer.__getitem__ = PyTorchDataSetTransformer.__getitem__

        mock_apply_transformation = MagicMock()
        monkeypatch.setattr(
            'datasets.pytorch_dataset_transformer.apply_transformation',
            mock_apply_transformation
        )

        sample = dataset_transformer[10]
        assert sample == {'element': 10, 'label': 1}
        assert not mock_apply_transformation.call_count

        mock_apply_transformation.return_value = 10
        dataset_transformer.transformations = [
            (0, {'sample_keys': ['element']}),
            (1, {'sample_keys': ['label']}),
            (2, {'sample_keys': ['element']})
        ]
        sample = dataset_transformer[10]
        assert sample == 10
        assert mock_apply_transformation.call_count == 3
        mock_apply_transformation.assert_any_call(
            0, {'element': 10, 'label': 1}, ['element'], {}
        )
        mock_apply_transformation.assert_any_call(1, 10, ['label'], {})
        mock_apply_transformation.assert_called_with(2, 10, ['element'], {})
