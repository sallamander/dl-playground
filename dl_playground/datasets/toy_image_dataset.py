"""Toy dataset for verifying / validating new network implementations"""

from itertools import product
import operator

import numpy as np
import skimage.draw
from torch.utils.data import Dataset

from utils.generic_utils import validate_config

OBJECT_COLORS = {
    'red': (1, 0, 0), 'orange': (1, 0.6471, 0), 'yellow': (1, 1, 0),
    'green': (0, 0.5, 0), 'blue': (0, 0, 1), 'indigo': (0.2941, 0, 0.5098),
    'violet': (0.9333, 0.5098, 0.9333), 'white': (1, 1, 1), 'black': (0, 0, 0)
}
OBJECT_SHAPES = ['ellipse', 'line', 'rectangle', 'triangle']
OBJECT_SIZES = ['small', 'medium', 'large']

MAX_FRACTION_OFFSETS_DICT = {
    'small': 0.85, 'medium': 0.70, 'large': 0.55
}
SIZE_BOUNDS_DICT = {
    'small': (0.05, 0.15), 'medium': (0.20, 0.40), 'large': (0.50, 0.70)
}


def generate_ellipse_coordinates(image_shape, centerpoint, size_bin):
    """Generate coordinates for a randomly placed ellipse in an image

    This generates image coordinates to place a randomly generated ellipse
    within an image. The size of the ellipse is bounded by the `size_bin`
    argument, where 'small' bounds it to > 5% and <=15% of the `image_shape`,
    'medium' bounds it to >20% and <= 40% of the `image_shape`, and 'large'
    bounds it to >50% and <=70%, but randomly chosen within those bounds.

    :param image_shape: (height, width) of the image that the ellipse will be
     placed into
    :type image_shape: tuple(int)
    :param centerpoint: (y, x) coordinates denoting the center of the ellipse
    :type centerpoint: tuple(int)
    :param size_bin: size bin that the ellipse should fall in, one of
     `OBJECT_SIZES`
    :type size_bin: str
    :return: coordinates into a numpy.ndarray of `image_shape` that represent
     the randomly generated ellipse
    :rtype: tuple(numpy.ndarray)
    """

    size_bounds = SIZE_BOUNDS_DICT[size_bin]

    # divide by 2 because these will be used to calculate the radius, and the
    # `size_bounds` themselves are relative to the entire object (and the
    # radius only represents half of an ellipse's bounds)
    y_radius_min = image_shape[0] * size_bounds[0] // 2.
    y_radius_max = image_shape[0] * size_bounds[1] // 2.
    x_radius_min = image_shape[1] * size_bounds[0] // 2.
    x_radius_max = image_shape[1] * size_bounds[1] // 2.

    y_radius = np.random.randint(y_radius_min, y_radius_max)
    x_radius = np.random.randint(x_radius_min, x_radius_max)
    y_radius = max(y_radius, 2)
    x_radius = max(x_radius, 2)

    ellipse_coordinates = skimage.draw.ellipse(
        r=centerpoint[0], c=centerpoint[1],
        r_radius=y_radius, c_radius=x_radius,
        shape=image_shape
    )
    return ellipse_coordinates


def generate_line_coordinates(image_shape, centerpoint, size_bin):
    """Generate coordinates for a randomly placed line in an image

    This generates image coordinates to place a randomly generated line
    within an image. The size of the line is bounded by the `size_bin`
    argument, where 'small' bounds it to > 5% and <=15% of the `image_shape`,
    'medium' bounds it to >20% and <= 40% of the `image_shape`, and 'large'
    bounds it to >50% and <=70%, but randomly chosen within those bounds.

    :param image_shape: (height, width) of the image that the line will be
     placed into
    :type image_shape: tuple(int)
    :param centerpoint: (y, x) coordinates denoting the center of the line
    :type centerpoint: tuple(int)
    :param size_bin: size bin that the line should fall in, one of
     `OBJECT_SIZES`
    :type size_bin: str
    :return: coordinates into a numpy.ndarray of `image_shape` that represent
     the randomly generated line
    :rtype: tuple(numpy.ndarray)
    """

    size_bounds = SIZE_BOUNDS_DICT[size_bin]

    height_min = int(image_shape[0] * size_bounds[0])
    height_max = int(image_shape[0] * size_bounds[1])
    width_min = int(image_shape[1] * size_bounds[0])
    width_max = int(image_shape[1] * size_bounds[1])

    height = np.random.randint(height_min, height_max)
    width = np.random.randint(width_min, width_max)

    y_op_choices = np.random.choice(
        [operator.sub, operator.add], size=2, replace=False
    )
    x_op_choices = np.random.choice(
        [operator.sub, operator.add], size=2, replace=False
    )

    y_vertices = (
        int(y_op_choices[0](centerpoint[0], height // 2.)),
        int(y_op_choices[1](centerpoint[0], height // 2.))
    )
    x_vertices = (
        int(x_op_choices[0](centerpoint[0], width // 2.)),
        int(x_op_choices[1](centerpoint[0], width // 2.))
    )
    line_coordinates = skimage.draw.line(
        y_vertices[0], x_vertices[0],
        y_vertices[1], x_vertices[1]
    )

    return line_coordinates


def generate_rectangle_coordinates(image_shape, centerpoint, size_bin):
    """Generate coordinates for a randomly placed rectangle in an image

    This generates image coordinates to place a randomly generated rectangle
    within an image. The size of the rectangle is bounded by the `size_bin`
    argument, where 'small' bounds it to > 5% and <=15% of the `image_shape`,
    'medium' bounds it to >20% and <= 40% of the `image_shape`, and 'large'
    bounds it to >50% and <=70%, but randomly chosen within those bounds.

    :param image_shape: (height, width) of the image that the rectangle will be
     placed into
    :type image_shape: tuple(int)
    :param centerpoint: (y, x) coordinates denoting the center of the rectangle
    :type centerpoint: tuple(int)
    :param size_bin: size bin that the rectangle should fall in, one of
     `OBJECT_SIZES`
    :type size_bin: str
    :return: coordinates into a numpy.ndarray of `image_shape` that represent
     the randomly generated rectangle
    :rtype: tuple(numpy.ndarray)
    """

    size_bounds = SIZE_BOUNDS_DICT[size_bin]

    height_min = int(image_shape[0] * size_bounds[0])
    height_max = int(image_shape[0] * size_bounds[1])
    width_min = int(image_shape[1] * size_bounds[0])
    width_max = int(image_shape[1] * size_bounds[1])

    height = np.random.randint(height_min, height_max)
    width = np.random.randint(width_min, width_max)

    y_vertices = (
        centerpoint[0] - height // 2,
        centerpoint[0] - height // 2,
        centerpoint[0] + height // 2,
        centerpoint[0] + height // 2
    )
    x_vertices = (
        centerpoint[1] - width // 2,
        centerpoint[1] + width // 2,
        centerpoint[1] + width // 2,
        centerpoint[1] - width // 2
    )
    rectangle_coordinates = skimage.draw.polygon(
        y_vertices, x_vertices, shape=image_shape
    )

    return rectangle_coordinates


def generate_triangle_coordinates(image_shape, centerpoint, size_bin):
    """Generate coordinates for a randomly placed triangle in an image

    This generates image coordinates to place a randomly generated triangle
    within an image. The size of the ellipse is bounded by the `size_bin`
    argument, where 'small' bounds it to > 5% and <=15% of the `image_shape`,
    'medium' bounds it to >20% and <= 40% of the `image_shape`, and 'large'
    bounds it to >50% and <=70%, but randomly chosen within those bounds.

    :param image_shape: (height, width) of the image that the triangle will be
     placed into
    :type image_shape: tuple(int)
    :param centerpoint: (y, x) coordinates denoting the center of the triangle
    :type centerpoint: tuple(int)
    :param size_bin: size bin that the ellipse should fall in, one of
     `OBJECT_SIZES`
    :type size_bin: str
    :return: coordinates into a numpy.ndarray of `image_shape` that represent
     the randomly generated triangle
    :rtype: tuple(numpy.ndarray)
    """

    size_bounds = SIZE_BOUNDS_DICT[size_bin]

    # divide by 2 because these will be used to calculate the radius of the
    # circle surrounding the triangle, and the `size_bounds` themselves are
    # relative to the entire object (and the radius only represents half of an
    # circle's bounds)
    y_min = image_shape[0] * size_bounds[0] // 2.
    y_max = image_shape[0] * size_bounds[1] // 2.
    x_min = image_shape[1] * size_bounds[0] // 2.
    x_max = image_shape[1] * size_bounds[1] // 2.

    y_vertex1 = centerpoint[0] + np.random.randint(y_min, y_max)
    x_vertex1 = centerpoint[1] + np.random.randint(x_min, x_max)

    y_vertex2 = centerpoint[0] - np.random.randint(y_min, y_max)
    x_vertex2 = centerpoint[1] - np.random.randint(x_min, x_max)

    y_vertex3 = centerpoint[0] - np.random.randint(y_min, y_max)
    x_vertex3 = centerpoint[1] + np.random.randint(x_min, x_max)

    triangle_coordinates = skimage.draw.polygon(
        [y_vertex1, y_vertex2, y_vertex3], [x_vertex1, x_vertex2, x_vertex3],
        shape=image_shape
    )
    return triangle_coordinates


class ToyImageDataSet(Dataset):
    """ToyImageDataSet

    This dataset is intended to be used as a means to verify new network
    implementations to ensure that they are able to generalize. It produces
    input / target pairs for several tasks that should be easy for a network to
    learn. Since each input / target pair is randomly generated, each pair is
    effectively a "new" example for the network, and thus the input / target
    pairs from a single ToyImageDataSet object can be treated as coming from
    the training, validation, or test sets.

    Currently, the following types of tasks are supported:
    - Classification (for arbitrarily many classes up to 108), where the
      network needs to classify the target type, which is defined by the color
      (see OBJECT_COLORS), shape (see OBJECT_SHAPES), and size (see
      OBJECT_SIZES) of the object in the input image
    """

    required_config_keys = {'height', 'width', 'n_classes'}

    def __init__(self, config, size=None):
        """Init

        `config` must contain the following keys:
        - int height: height of the images to create
        - int width: width of the images to create
        - int n_classes: number of classes of target to use when creating the
          data

        It can additionally contain the following keys:
        - list[str] object_colors: colors to use for the object in each
          image; options include those in OBJECT_COLORS; defaults to
          OBJECT_COLORS
        - list[str] object_shapes: shapes to use for the object in each
          image; options include those in OBJECT_SHAPES; defaults to
          OBJECT_SHAPES
        - list[str] object_sizes: sizes to use for the object in each image;
          options include those in OBJECT_SIZES; defaults to OBJECT_SIZES

        :param config: specifies the configuration of the image, label pairs to
         generate as part of the dataset
        :param config: dict
        :param size: number of elements to return from the dataset before it is
         exhausted (i.e. before it is iterated over one time to completion),
         defaults to the number of unique combinations of object_colors,
         object_shapes, and object_sizes specified iin the config
        :type size: int
        """

        validate_config(config, self.required_config_keys)

        self.height = config['height']
        self.width = config['width']
        self.n_classes = config['n_classes']

        object_colors = config.get('object_colors', list(OBJECT_COLORS.keys()))
        # sort so that the order is always the same
        object_colors = sorted(object_colors)
        object_shapes = config.get('object_shapes', OBJECT_SHAPES)
        object_sizes = config.get('object_sizes', OBJECT_SIZES)

        msg = '{} contains an invalid option. Valid options are: {}'
        if set(object_colors).difference(OBJECT_COLORS):
            raise ValueError(msg.format('object_colors', OBJECT_COLORS))
        if set(object_shapes).difference(OBJECT_SHAPES):
            raise ValueError(msg.format('object_shapes', OBJECT_SHAPES))
        if set(object_sizes).difference(object_sizes):
            raise ValueError(msg.format('object_sizes', OBJECT_SIZES))

        self.object_colors = object_colors
        self.object_shapes = object_shapes
        self.object_sizes = object_sizes
        object_spec_options = list(product(
            object_colors, object_shapes, object_sizes
        ))[:self.n_classes]
        self.object_spec_options = {
            idx_spec: object_spec
            for idx_spec, object_spec in enumerate(object_spec_options)
        }

        if size is None:
            size = len(self.object_spec_options)
        self.size = size

    def __getitem__(self, _):
        """Return an randomly generated image, label pair

        :param _: unused
        :type _: int
        :return: dict with keys:
        - numpy.ndarray image: pixel data that contains a fuzzy background and
          a generated object in the foreground
        - int label: class label assigned to the returned image, corresponding
          to the color, shape, and size of the object in the foreground
        :rtype: dict
        """

        random_array = np.random.random((self.height, self.width))
        random_array = np.expand_dims(random_array, axis=-1)
        zeros = np.zeros_like(random_array)
        image = np.concatenate([random_array, zeros, zeros], axis=-1)

        idx_object_spec = np.random.choice(len(self.object_spec_options))
        object_color, object_shape, object_size = (
            self.object_spec_options[idx_object_spec]
        )
        object_slices = self._get_object_coordinates(
            object_shape, object_size
        )
        object_values = OBJECT_COLORS[object_color]
        for idx_channel, channel_value in enumerate(object_values):
            image[object_slices[0], object_slices[1], idx_channel] = (
                channel_value
            )

        return (image, idx_object_spec)

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return self.size

    def _get_object_coordinates(self, object_shape, object_size):
        """Return coordinates to add an object to an image

        :param object_shape: shape of the object, one of self.object_shapes
        :type object_shape: str
        :param object_size: size of the object, one of self.object_sizes
        :type object_size: str
        :return: coordinates of the pixels that comprise an object to add to an
         image
        :rtype: tuple(numpy.ndarray)
        """

        max_fraction_offset = MAX_FRACTION_OFFSETS_DICT[object_size]
        image_shape = (self.height, self.width)
        max_y_offset = int(image_shape[0] / 2. * max_fraction_offset)
        max_x_offset = int(image_shape[1] / 2. * max_fraction_offset)
        center_offset_y = np.random.randint(-max_y_offset, max_y_offset)
        center_offset_x = np.random.randint(-max_x_offset, max_x_offset)
        centerpoint = (
            image_shape[0] // 2. + center_offset_y,
            image_shape[1] // 2. + center_offset_x
        )

        if object_shape == 'ellipse':
            coordinates_fn = generate_ellipse_coordinates
        elif object_shape == 'rectangle':
            coordinates_fn = generate_rectangle_coordinates
        elif object_shape == 'line':
            coordinates_fn = generate_line_coordinates
        elif object_shape == 'triangle':
            coordinates_fn = generate_triangle_coordinates

        object_coordinates = coordinates_fn(
            image_shape=(self.height, self.width), centerpoint=centerpoint,
            size_bin=object_size
        )
        object_coordinates = (
            np.minimum(object_coordinates[0], image_shape[0] - 1),
            np.minimum(object_coordinates[1], image_shape[1] - 1)
        )
        return object_coordinates
