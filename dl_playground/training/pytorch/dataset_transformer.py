"""Transformer for transformations of a torch.utils.data.Dataset"""

from torch.utils.data import Dataset

from datasets.ops import apply_transformation


class PyTorchDataSetTransformer(Dataset):
    """Transformer to apply transformations to a torch.utils.data.Dataset"""

    def __init__(self, numpy_dataset, transformations=None):
        """Init

        :param numpy_dataset: dataset that provies samples for training
        :type numpy_dataset: torch.utils.data.Dataset object
        :param transformations: holds 2 element tuples with the first element
         being a function to apply to the dataset samples and the second
         element being a dictionary of keyword arguments to pass to those
         functions
        :type transformations: list[tuple(function, dict)]
        """

        self.numpy_dataset = numpy_dataset
        self.transformations = transformations or []

    def __getitem__(self, idx):
        """Return the transformed sample at index `idx` of `self.numpy_dataset`

        :return: sample returned from `self.numpy_dataset.__getitem__`
        :rtype: dict
        """

        sample = self.numpy_dataset[idx]

        it = self.transformations
        for transformation_fn, transformation_fn_kwargs in it:
            transformation_fn_kwargs = transformation_fn_kwargs.copy()
            sample_keys = transformation_fn_kwargs.pop('sample_keys')

            sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )

        inputs = [
            sample[input_key] for input_key in self.numpy_dataset.input_keys
        ]
        targets = [
            sample[target_key] for target_key in self.numpy_dataset.target_keys
        ]

        if len(inputs) == 1:
            inputs = inputs[0]
        if len(targets) == 1:
            targets = targets[0]

        return inputs, targets

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.numpy_dataset)
