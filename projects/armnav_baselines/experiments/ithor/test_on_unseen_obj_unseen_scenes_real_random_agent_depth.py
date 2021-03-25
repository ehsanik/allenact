import ai2thor
import gym

from plugins.ithor_arm_plugin.ithor_arm_constants import ENV_ARGS
from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor, DepthSensorThor
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import RandomAgentWDoneActionTaskSampler, WDoneActionTaskSampler
from plugins.ithor_arm_plugin.ithor_arm_tasks import PickUpDropOffTask, WDoneActionTask
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from projects.armnav_baselines.experiments.ithor.armnav_ithor_base import (
    ArmNaviThorBaseConfig,
)
from projects.armnav_baselines.experiments.armnav_mixin_ddppo import (
    ArmNavMixInPPOConfig,
)
from projects.armnav_baselines.experiments.armnav_mixin_simplegru import (
    ArmNavMixInSimpleGRUConfig,
)
import torch.nn as nn

from projects.armnav_baselines.experiments.ithor.real_random_agent_depth import RealDepthRandomAgentLocArmNav
from projects.armnav_baselines.models.arm_nav_models import ArmNavBaselineActorCritic


class TestOnUObjUSceneRealDepthRandomAgentLocArmNav(
    RealDepthRandomAgentLocArmNav
):

    MAX_STEPS = 200
    TASK_SAMPLER = RandomAgentWDoneActionTaskSampler
    TEST_SCENES = ArmNaviThorBaseConfig.TEST_SCENES
    OBJECT_TYPES = ArmNaviThorBaseConfig.UNSEEN_OBJECT_TYPES
    # TASK_SAMPLER  =WDoneActionTaskSampler
    # remove
    # VISUALIZE = True
    # NUM_PROCESSES = 1
    # NUMBER_OF_TEST_PROCESS = 1