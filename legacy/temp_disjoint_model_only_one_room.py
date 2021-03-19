from torch import nn

from core.base_abstractions.sensor import RGBSensor
from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor
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


class TmpOnly1RoomDisjointModelArmNaviThorRGBPPOExperimentConfig(
    ArmNaviThorBaseConfig, ArmNavMixInPPOConfig, ArmNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
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

    TRAIN_SCENES = ['FloorPlan1_physics']
    TEST_SCENES = ['FloorPlan1_physics']
    VALID_SCENES = ['FloorPlan1_physics']


    NUM_PROCESSES = 25


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        return DisjointArmNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(PickUpDropOffTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            # goal_sensor_uuid=goal_sensor_uuid,
            # rgb_uid="rgb_uid" if has_rgb else None,
            # depth_uid="depth_uid" if has_depth else None,
            hidden_size=512,
            # goal_dims=32,
        )


    @classmethod
    def tag(cls):
        return cls.__name__
