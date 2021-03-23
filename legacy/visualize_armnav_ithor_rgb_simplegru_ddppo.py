from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor
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
from projects.armnav_baselines.experiments.ithor.armnav_ithor_rgb_simplegru_ddppo import ArmNaviThorRGBPPOExperimentConfig


class VisualizeArmNaviThorRGBPPOExperimentConfig(
    ArmNaviThorRGBPPOExperimentConfig
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

    MAX_STEPS = 200
    # TEST_SCENES = ArmNaviThorRGBPPOExperimentConfig.TRAIN_SCENES
    VISUALIZE = True
    NUM_PROCESSES = 1
    NUMBER_OF_TEST_PROCESS = 1


    @classmethod
    def tag(cls):
        return cls.__name__
