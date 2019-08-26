"""Unit tests for datasets.augmented_dataset.py"""

from unittest.mock import MagicMock

import torch

from datasets.augmented_dataset import AugmentedDataset


class TestAugmentedDataset(object):
    """Tests for AugmentedDataset"""

    def test_init(self):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.
        """

        numpy_dataset = MagicMock()
        augmented_dataset = AugmentedDataset(
            numpy_dataset=numpy_dataset
        )
        assert id(augmented_dataset.numpy_dataset) == id(numpy_dataset)
        assert not augmented_dataset.transformations

        transformations = MagicMock()
        augmented_dataset = AugmentedDataset(
            numpy_dataset=numpy_dataset, transformations=transformations
        )
        assert id(augmented_dataset.transformations) == id(transformations)

    def test_getitem(self, monkeypatch):
        """Test __getitem__ method"""

        def mock_getitem(self, idx):
            """Mock __getitem__ method for a numpy_datset"""
            return {'element': idx, 'label': 1}

        augmented_dataset = MagicMock()
        augmented_dataset.numpy_dataset = MagicMock()
        augmented_dataset.numpy_dataset.input_keys = ['element']
        augmented_dataset.numpy_dataset.target_keys = ['label']
        augmented_dataset.numpy_dataset.__getitem__ = mock_getitem
        augmented_dataset.__getitem__ = AugmentedDataset.__getitem__

        mock_apply_transformation = MagicMock()
        monkeypatch.setattr(
            'datasets.augmented_dataset.apply_transformation',
            mock_apply_transformation
        )

        element, label = augmented_dataset[10]
        assert not mock_apply_transformation.call_count

        mock_apply_transformation.return_value = {
            'element': 2, 'label': 10
        }
        augmented_dataset.transformations = [
            (0, {'sample_keys': ['element']}),
            (1, {'sample_keys': ['label']}),
            (2, {'sample_keys': ['element']})
        ]
        sample = augmented_dataset[10]
        assert sample['element'] == 2
        assert sample['label'] == 10
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

        augmented_dataset = MagicMock()
        augmented_dataset.__len__ = AugmentedDataset.__len__
        augmented_dataset.numpy_dataset = MagicMock()

        mock_len = MagicMock()
        augmented_dataset.numpy_dataset.__len__ = mock_len

        for len_value in [1, 3, 5]:
            mock_len.return_value = len_value
            assert len(augmented_dataset) == len_value

    def test_as_generator(self):
        """Test `as_generator` method

        This tests two cases:
        - `shuffle=False`
        - `shuffle=True`

        It only tests when `n_workers=0`, deferring the testing of
        `n_workers=1` to integration tests.
        """

        def mock_get_item(self, idx):
            """Mock __getitem__ magic method"""
            return {idx: idx}
        
        def mock_len(self):
            """Mock __len__ magic method"""
            return 9

        augmented_dataset = MagicMock()
        augmented_dataset.__getitem__ = mock_get_item
        augmented_dataset.__len__ = mock_len
        augmented_dataset.as_generator = AugmentedDataset.as_generator

        # set the seed to one where we know that the shuffle won't by random
        # chance produce an unshuffled version
        torch.manual_seed(1)
        for shuffle in [True, False]:
            generator = augmented_dataset.as_generator(
                self=augmented_dataset, n_workers=0, shuffle=shuffle
            )
            samples = [next(generator) for _ in range(9)]
            indices, values = [], []
            for sample in samples:
                for idx, value in sample.items():
                    indices.append(idx)
                    values.append(value.tolist())

            if shuffle:
                assert indices != list(range(9))
                assert values != list(range(9))
            else:
                assert indices == list(range(9))
                assert values == list(range(9))
