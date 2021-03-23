import ai2thor
import gym

from plugins.ithor_arm_plugin.ithor_arm_constants import ENV_ARGS
from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor, DepthSensorThor
from plugins.ithor_arm_plugin.ithor_arm_tasks import PickUpDropOffTask
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

from projects.armnav_baselines.models.arm_nav_models import ArmNavBaselineActorCritic


class DepthRGBArmNaviThorRGBPPOExperimentConfig(
    ArmNaviThorBaseConfig, ArmNavMixInPPOConfig, ArmNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        DepthSensorThor(
            height=ArmNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        RGBSensorThor(
            height=ArmNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        # GoalObjectTypeThorSensor(object_types=ArmNaviThorBaseConfig.OBJECT_TYPES,),
        RelativeAgentArmToObjectSensor(),
        RelativeObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    MAX_STEPS = 200


    def __init__(self):
        super().__init__()

        assert self.CAMERA_WIDTH == 224 and self.CAMERA_HEIGHT == 224 and self.VISIBILITY_DISTANCE == 1 and self.STEP_SIZE == 0.25
        self.ENV_ARGS = {**ENV_ARGS,  'renderDepthImage':True}
        # self.ENV_ARGS = dict(
        #     width=self.CAMERA_WIDTH,
        #     height=self.CAMERA_HEIGHT,
        #     visibilityDistance=self.VISIBILITY_DISTANCE,
        #     gridSize=self.STEP_SIZE,
        #     snapToGrid=False,
        #     agentMode='arm',
        #     agentControllerType='mid-level',
        #     useMassThreshold = True, massThreshold = 10,
        #     include_private_scenes=False,
        #     server_class=ai2thor.fifo_server.FifoServer,
        #     renderDepthImage=True,
        # )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ArmNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(PickUpDropOffTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

    @classmethod
    def tag(cls):
        return cls.__name__
