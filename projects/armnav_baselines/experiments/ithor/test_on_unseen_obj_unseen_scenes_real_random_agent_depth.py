
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import RandomAgentWDoneActionTaskSampler

from projects.armnav_baselines.experiments.ithor.armnav_ithor_base import (
    ArmNaviThorBaseConfig,
)

from projects.armnav_baselines.experiments.ithor.armnav_depth import ArmNavDepth


class TestOnUObjUSceneRealDepthRandomAgentLocArmNav(
    ArmNavDepth
):

    MAX_STEPS = 200
    TASK_SAMPLER = RandomAgentWDoneActionTaskSampler
    TEST_SCENES = ArmNaviThorBaseConfig.TEST_SCENES
    OBJECT_TYPES = ArmNaviThorBaseConfig.UNSEEN_OBJECT_TYPES
