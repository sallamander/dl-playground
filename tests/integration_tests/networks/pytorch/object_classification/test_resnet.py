"""Integration tests for networks.pytorch.object_classification.resnet"""

from networks.pytorch.object_classification.resnet import ResNet


class TestResNet(object):
    """Tests for ResNet"""

    def test_init(self):
        """Test __init__

        This simply tests that an instantiated ResNet has the expected number
        of parameter objects returned from its `parameters()`. This is more or
        less a check simply to make sure that all of the BottleNeck and
        ProjectionShortcut modules parameters are included in the ResNet
        itself.

        The following parameters are expected:
        - From the ResNet itself (excluding BottleneckBlock or
          ProjectionShortcut objects): 6 (2 for conv1 plus bias, 2 for bn plus
          bias, 2 for final linear layer plus bias)
        - From each BottleneckBlock: 9
        - From each ProjectionShortcut: 12
        """

        test_cases = [
            {'n_initial_channels': 64, 'n_blocks_per_stage': [3, 4, 6, 3],
             'n_expected_objects': 162},
            {'n_initial_channels': 64, 'n_blocks_per_stage': [3, 4],
             'n_expected_objects': 75},
            {'n_initial_channels': 64, 'n_blocks_per_stage': [3, 4, 23, 3],
             'n_expected_objects': 315}
        ]

        for test_case in test_cases:
            n_expected_objects = test_case.pop('n_expected_objects')
            test_case['n_channels'] = 3
            test_case['n_classes'] = 1000
            test_case['version'] = 'original'

            resnet = ResNet(test_case)
            assert len(list(resnet.parameters())) == n_expected_objects
