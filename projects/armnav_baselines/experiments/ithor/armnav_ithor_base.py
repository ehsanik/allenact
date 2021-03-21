import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from plugins.ithor_arm_plugin.ithor_arm_constants import TRAIN_OBJECTS
from projects.armnav_baselines.experiments.armnav_thor_base import (
    ArmNavThorBaseConfig,
)


class ArmNaviThorBaseConfig(ArmNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    NUM_PROCESSES = 40
    # add all the arguments here
    TOTAL_NUMBER_SCENES = 30

    TRAIN_SCENES = ["FloorPlan{}_physics".format(str(i)) for i in range(1, TOTAL_NUMBER_SCENES + 1) if (i % 3 == 1 or i % 3 == 0) and i != 28] # last scenes are really bad
    TEST_SCENES = ["FloorPlan{}_physics".format(str(i)) for i in range(1, TOTAL_NUMBER_SCENES + 1) if i % 3 == 2 and i % 6 == 2]
    VALID_SCENES = ["FloorPlan{}_physics".format(str(i)) for i in range(1, TOTAL_NUMBER_SCENES + 1) if i % 3 == 2 and i % 6 == 5]

    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert len(ALL_SCENES) == TOTAL_NUMBER_SCENES - 1 and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES - 1



    OBJECT_TYPES = tuple(
        sorted(
            TRAIN_OBJECTS
        )
    )

    # #
    # TRAIN_SCENES = TEST_SCENES = VALID_SCENES = ['FloorPlan1_physics']
    # OBJECT_TYPES = tuple(
    #     ['Apple']
    # )
