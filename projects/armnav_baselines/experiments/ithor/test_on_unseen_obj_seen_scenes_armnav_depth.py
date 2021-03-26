
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import RandomAgentWDoneActionTaskSampler, WDoneActionTaskSampler

from projects.armnav_baselines.experiments.ithor.armnav_ithor_base import (
    ArmNaviThorBaseConfig,
)

from projects.armnav_baselines.experiments.ithor.armnav_depth import ArmNavDepth


class TestOnUObjSSceneArmNavDepth(
    ArmNavDepth
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""


    MAX_STEPS = 200
    TASK_SAMPLER = RandomAgentWDoneActionTaskSampler
    TEST_SCENES = ArmNaviThorBaseConfig.TRAIN_SCENES
    OBJECT_TYPES = ArmNaviThorBaseConfig.UNSEEN_OBJECT_TYPES
