import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.armnav_baselines.experiments.armnav_thor_base import (
    ArmNavThorBaseConfig,
)


class ArmNaviThorBaseConfig(ArmNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    NUM_PROCESSES = 40
    #TODO add all the arguments here
    #TODO add train, test, val scenes here


    OBJECT_TYPES = tuple(
        sorted(
            [
                'Apple', 'Bread', 'Tomato', 'Lettuce', 'Pot', 'Mug'
            ]
        )
    )
