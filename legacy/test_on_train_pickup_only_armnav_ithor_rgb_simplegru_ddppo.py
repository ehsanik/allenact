from plugins.ithor_arm_plugin.ithor_arm_sensors import RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor
from plugins.ithor_arm_plugin.ithor_arm_task_samplers import OnlyPickupGeneralSampler
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from projects.armnav_baselines.experiments.ithor.armnav_ithor_base import (
    ArmNaviThorBaseConfig,
)
from legacy.pickup_only_armnav_ithor_rgb_simplegru_ddppo import PickUpOnlyArmNaviThorRGBPPOExperimentConfig


class TestTrainPickUpOnlyArmNaviThorRGBPPOExperimentConfig(
    PickUpOnlyArmNaviThorRGBPPOExperimentConfig
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
    TASK_SAMPLER = OnlyPickupGeneralSampler
    MAX_STEPS = 120

    TEST_SCENES = PickUpOnlyArmNaviThorRGBPPOExperimentConfig.TRAIN_SCENES
    VISUALIZE = True
    NUM_PROCESSES = 1
    NUMBER_OF_TEST_PROCESS = 1

    @classmethod
    def tag(cls):
        return cls.__name__
