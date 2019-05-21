"""Unit tests for networks.pytorch.object_classification.resnet"""

from unittest.mock import MagicMock

import networks.pytorch.object_classification.resnet as pytorch_resnet

BottleneckBlock = pytorch_resnet.BottleneckBlock
ProjectionShortcut = pytorch_resnet.ProjectionShortcut
ResNet = pytorch_resnet.ResNet


def check_forward(object_class, attribute_call_counts, monkeypatch):
    """Assert attribute call counts are met when calling object_class.forward

    This function creates a MagicMock of `object_class`, and additionally
    creates MagicMock objects for all attributes in `attribute_call_counts` to
    assign to the MagicMock for `object_class`. It then calls
    `object_class.forward`, and verifies that each of the attributes is called
    the expected number of times. It additional checks that torch.add is called
    once (since this is used to test residual type object classes).

    :param object_class: object class whose forward method will be tested
    :type object_class: class
    :param attribute_call_counts: names of attributes and their expected call
     counts after running `object_class.forward`
    :type attribute_call_counts: dict
    :param monkeypatch: monkeypatch object
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    """

    module = MagicMock()
    mock_attributes = {}
    for attr_name in attribute_call_counts:
        mock_attribute = MagicMock()
        setattr(module, attr_name, mock_attribute)
        mock_attributes[attr_name] = mock_attribute

    mock_add = MagicMock()
    monkeypatch.setattr('torch.add', mock_add)

    module.forward = object_class.forward
    module.forward(module, 'sentinel')

    for attr_name in attribute_call_counts:
        mock_attribute = mock_attributes[attr_name]
        assert (
            mock_attribute.call_count ==
            attribute_call_counts[attr_name]
        )
    assert mock_add.call_count == 1


class TestBottleneckBlock(object):
    """Tests for BottleneckBlock"""

    def test_init(self):
        """Test __init__

        This tests a couple of things:
        - The `torch.nn` operations used in the BottleneckBlock are set as
          attributes on the object
        - The `torch.nn` operations used in the BottleneckBlock are set as
          expected (e.g. with the right number of filters and such)
        """

        test_cases = [
            {'n_in_channels': 4, 'n_out_channels': 8,
             'conv1_in_channels': 4, 'conv1_out_channels': 8,
             'conv1_kernel_size': (1, 1), 'conv1_stride': (1, 1),
             'conv1_bias': None, 'bn1_num_features': 8,
             'conv2_in_channels': 8, 'conv2_out_channels': 8,
             'conv2_kernel_size': (3, 3), 'conv2_stride': (1, 1),
             'conv2_bias': None, 'bn2_num_features': 8,
             'conv3_in_channels': 8, 'conv3_out_channels': 32,
             'conv3_kernel_size': (1, 1), 'conv3_stride': (1, 1),
             'conv3_bias': None, 'bn3_num_features': 32},
            {'n_in_channels': 512, 'n_out_channels': 128,
             'conv1_in_channels': 512, 'conv1_out_channels': 128,
             'conv1_kernel_size': (1, 1), 'conv1_stride': (1, 1),
             'conv1_bias': None, 'bn1_num_features': 128,
             'conv2_in_channels': 128, 'conv2_out_channels': 128,
             'conv2_kernel_size': (3, 3), 'conv2_stride': (1, 1),
             'conv2_bias': None, 'bn2_num_features': 128,
             'conv3_in_channels': 128, 'conv3_out_channels': 512,
             'conv3_kernel_size': (1, 1), 'conv3_stride': (1, 1),
             'conv3_bias': None, 'bn3_num_features': 512}
        ]

        conv_attributes = [
            'in_channels', 'out_channels', 'kernel_size', 'stride', 'bias'
        ]

        for test_case in test_cases:
            block = BottleneckBlock(
                n_in_channels=test_case['n_in_channels'],
                n_out_channels=test_case['n_out_channels']
            )

            for conv_name in ['conv1', 'conv2', 'conv3']:
                conv = getattr(block, conv_name)
                for conv_attr in conv_attributes:
                    attr_value = getattr(conv, conv_attr)
                    test_case_name = '{}_{}'.format(conv_name, conv_attr)
                    assert attr_value == test_case[test_case_name]
            for bn_name in ['bn1', 'bn2', 'bn3']:
                bn = getattr(block, bn_name)
                test_case_name = '{}_num_features'.format(bn_name)
                assert bn.num_features == test_case[test_case_name]

            assert block.relu is not None

    def test_forward(self, monkeypatch):
        """Test forward method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        attribute_call_counts = {
            'conv1': 1, 'bn1': 1, 'relu': 3,
            'conv2': 1, 'bn2': 1, 'conv3': 1, 'bn3': 1
        }
        check_forward(BottleneckBlock, attribute_call_counts, monkeypatch)


class TestProjectionShortcut(object):
    """Tests for ProjectionShortcut"""

    def test_init(self):
        """Test __init__

        This tests a couple of things:
        - The `torch.nn` operations used in the ProjectionShortcut are set as
          attributes on the object
        - The `torch.nn` operations used in the ProjectionShortcut are set as
          expected (e.g. with the right number of filters and such)
        """

        test_cases = [
            {'n_in_channels': 4, 'n_out_channels': 8, 'stride': (1, 1),
             'conv1_in_channels': 4, 'conv1_out_channels': 8,
             'conv1_kernel_size': (1, 1), 'conv1_stride': (1, 1),
             'conv1_bias': None, 'bn1_num_features': 8,
             'conv2_in_channels': 8, 'conv2_out_channels': 8,
             'conv2_kernel_size': (3, 3), 'conv2_stride': (1, 1),
             'conv2_bias': None, 'bn2_num_features': 8,
             'conv3_in_channels': 8, 'conv3_out_channels': 32,
             'conv3_kernel_size': (1, 1), 'conv3_stride': (1, 1),
             'conv3_bias': None, 'bn3_num_features': 32,
             'projection_conv_in_channels': 4,
             'projection_conv_out_channels': 32,
             'projection_conv_kernel_size': (1, 1),
             'projection_conv_stride': (1, 1), 'projection_conv_bias': None,
             'projection_bn_num_features': 32},
            {'n_in_channels': 512, 'n_out_channels': 128, 'stride': (2, 2),
             'conv1_in_channels': 512, 'conv1_out_channels': 128,
             'conv1_kernel_size': (1, 1), 'conv1_stride': (2, 2),
             'conv1_bias': None, 'bn1_num_features': 128,
             'conv2_in_channels': 128, 'conv2_out_channels': 128,
             'conv2_kernel_size': (3, 3), 'conv2_stride': (1, 1),
             'conv2_bias': None, 'bn2_num_features': 128,
             'conv3_in_channels': 128, 'conv3_out_channels': 512,
             'conv3_kernel_size': (1, 1), 'conv3_stride': (1, 1),
             'conv3_bias': None, 'bn3_num_features': 512,
             'projection_conv_in_channels': 512,
             'projection_conv_out_channels': 512,
             'projection_conv_kernel_size': (1, 1),
             'projection_conv_stride': (2, 2), 'projection_conv_bias': None,
             'projection_bn_num_features': 512}
        ]

        conv_attributes = [
            'in_channels', 'out_channels', 'kernel_size', 'stride', 'bias'
        ]

        for test_case in test_cases:
            block = ProjectionShortcut(
                n_in_channels=test_case['n_in_channels'],
                n_out_channels=test_case['n_out_channels'],
                stride=test_case['stride']
            )

            for conv_name in ['conv1', 'conv2', 'conv3', 'projection_conv']:
                conv = getattr(block, conv_name)
                for conv_attr in conv_attributes:
                    attr_value = getattr(conv, conv_attr)
                    test_case_name = '{}_{}'.format(conv_name, conv_attr)
                    assert attr_value == test_case[test_case_name]
            for bn_name in ['bn1', 'bn2', 'bn3', 'projection_bn']:
                bn = getattr(block, bn_name)
                test_case_name = '{}_num_features'.format(bn_name)
                assert bn.num_features == test_case[test_case_name]

            assert block.relu is not None

    def test_forward(self, monkeypatch):
        """Test forward method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        attribute_call_counts = {
            'conv1': 1, 'bn1': 1, 'relu': 3,
            'conv2': 1, 'bn2': 1, 'conv3': 1, 'bn3': 1,
            'projection_conv': 1, 'projection_bn': 1
        }
        check_forward(ProjectionShortcut, attribute_call_counts, monkeypatch)


class TestResNet(object):
    """Tests for ResNet"""

    def test_set_layers(self, monkeypatch):
        """Test _set_layers

        This tests a couple of things:
        - The `torch.nn` operations used in the ResNet._set_layers are set as
          attributes on the object
        - The `torch.nn` operations used in the ResNet._set_layers are set as
          expected (e.g. with the right number of filters and such)
        """

        test_cases = [
            {'n_initial_channels': 4, 'n_blocks_per_stage': [2, 3],
             'conv_in_channels': 3, 'conv_out_channels': 4,
             'conv_kernel_size': (7, 7), 'conv_stride': (2, 2),
             'conv_padding': (3, 3), 'conv_bias': True,
             'bn_num_features': 4, 'max_pooling_kernel_size': (3, 3),
             'max_pooling_stride': (2, 2), 'n_projection_shortcut_calls': 2,
             'n_bottleneck_block_calls': 3, 'linear_in_features': 32},
            {'n_initial_channels': 64, 'n_blocks_per_stage': [3, 4, 6, 3],
             'conv_in_channels': 3, 'conv_out_channels': 64,
             'conv_kernel_size': (7, 7), 'conv_stride': (2, 2),
             'conv_padding': (3, 3), 'conv_bias': True,
             'bn_num_features': 64, 'max_pooling_kernel_size': (3, 3),
             'max_pooling_stride': (2, 2), 'n_projection_shortcut_calls': 4,
             'n_bottleneck_block_calls': 12, 'linear_in_features': 2048}
        ]

        conv_attributes = [
            'in_channels', 'out_channels', 'kernel_size', 'stride', 'bias',
            'padding'
        ]

        mock_projection_shorcut = MagicMock()
        mock_bottleneck_block = MagicMock()
        monkeypatch.setattr(
            ('networks.pytorch.object_classification.resnet'
             '.ProjectionShortcut'),
            mock_projection_shorcut
        )
        monkeypatch.setattr(
            ('networks.pytorch.object_classification.resnet'
             '.BottleneckBlock'),
            mock_bottleneck_block
        )

        for test_case in test_cases:
            resnet = MagicMock()
            resnet.config = {
                'n_initial_channels': test_case['n_initial_channels'],
                'n_blocks_per_stage': test_case['n_blocks_per_stage']
            }
            resnet._set_layers = ResNet._set_layers
            resnet._set_layers(self=resnet)

            # check the conv, bn, max pooling, average pooling, and linear
            # layers
            conv = getattr(resnet, 'conv')
            for conv_attr in conv_attributes:
                attr_value = getattr(conv, conv_attr)
                test_case_name = 'conv_{}'.format(conv_attr)
                if 'bias' in test_case_name:
                    assert attr_value is not None
                else:
                    assert attr_value == test_case[test_case_name]

            bn = getattr(resnet, 'bn')
            test_case_name = 'bn_num_features'
            assert bn.num_features == test_case[test_case_name]

            max_pooling = getattr(resnet, 'max_pooling')
            for pooling_attr in ['kernel_size', 'stride']:
                attr_value = getattr(max_pooling, pooling_attr)
                test_case_name = 'max_pooling_{}'.format(pooling_attr)
                assert attr_value == test_case[test_case_name]

            assert resnet.relu is not None
            assert resnet.average_pooling.kernel_size == (7, 7)
            assert resnet.linear.in_features == test_case['linear_in_features']
            assert resnet.linear.out_features == 1000

            # check to make sure the correct number of residual stages are
            # present, and that the consist of the correct blocks
            assert (len(resnet.residual_stages) ==
                    len(test_case['n_blocks_per_stage']))
            for idx_stage, stage in enumerate(resnet.residual_stages):
                assert len(stage) == test_case['n_blocks_per_stage'][idx_stage]
            assert (mock_projection_shorcut.call_count ==
                    test_case['n_projection_shortcut_calls'])
            assert (mock_bottleneck_block.call_count ==
                    test_case['n_bottleneck_block_calls'])

            # reset the call count for the next iteration of the loop
            mock_projection_shorcut.call_count = 0
            mock_bottleneck_block.call_count = 0

    def test_forward(self, monkeypatch):
        """Test forward method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        attribute_call_counts = {
            'conv': 1, 'bn': 1, 'relu': 1, 'max_pooling': 1,
            'average_pooling': 1, 'linear': 1
        }

        module = MagicMock()
        module.residual_stages = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock(), MagicMock()]
        ]
        mock_attributes = {}
        for attr_name in attribute_call_counts:
            mock_attribute = MagicMock()
            setattr(module, attr_name, mock_attribute)
            mock_attributes[attr_name] = mock_attribute

        mock_add = MagicMock()
        monkeypatch.setattr('torch.add', mock_add)

        module.forward = ResNet.forward
        module.forward(module, 'sentinel')

        for attr_name in attribute_call_counts:
            mock_attribute = mock_attributes[attr_name]
            assert (
                mock_attribute.call_count ==
                attribute_call_counts[attr_name]
            )
        for mock_residual_stage in module.residual_stages:
            for mock_residual_block in mock_residual_stage:
                assert mock_residual_block.call_count == 1
