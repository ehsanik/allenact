import gym

from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor, BlindSensorThor
from plugins.ithor_arm_plugin.ithor_arm_tasks import PickUpDropOffTask

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

from projects.armnav_baselines.models.arm_nav_models import ArmNavBaselineActorCritic


class BlindArmNaviThorRGBPPOExperimentConfig(
    ArmNaviThorBaseConfig, ArmNavMixInPPOConfig, ArmNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        BlindSensorThor(
            height=ArmNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid="rgb_lowres",
        ),
        # GoalObjectTypeThorSensor(object_types=ArmNaviThorBaseConfig.OBJECT_TYPES,),
        RelativeAgentArmToObjectSensor(),
        RelativeObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    #
    MAX_STEPS = 200

    @classmethod
    def tag(cls):
        return cls.__name__
