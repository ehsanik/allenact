from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from projects.armnav_baselines.experiments.ithor.armnav_ithor_base import (
    ArmNaviThorBaseConfig,
)
from projects.armnav_baselines.experiments.armnav_mixin_ddppo import (
    ArmNavMixInPPOConfig,
)
from projects.armnav_baselines.experiments.armnav_mixin_resnetgru import (
    ArmNavMixInResNetGRUConfig,
)


class ArmNaviThorRGBPPOExperimentConfig(
    ArmNaviThorBaseConfig, ArmNavMixInPPOConfig, ArmNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ArmNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres", #TODO
        ),
        GoalObjectTypeThorSensor(object_types=ArmNaviThorBaseConfig.TARGET_TYPES,),
    ]

    @classmethod
    def tag(cls):
        return cls.__name__
