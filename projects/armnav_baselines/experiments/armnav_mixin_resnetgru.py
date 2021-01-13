from typing import Sequence, Union

import gym
import torch.nn as nn
from torchvision import models

from core.base_abstractions.preprocessor import Preprocessor
from core.base_abstractions.preprocessor import ResNetPreprocessor
from core.base_abstractions.sensor import RGBSensor, DepthSensor
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.ithor_arm_plugin.ithor_arm_tasks import PickUpDropOffTask
from projects.armnav_baselines.experiments.armnav_base import ArmNavBaseConfig
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
) #TODO change
from projects.armnav_baselines.models.arm_nav_models import (
    ResnetTensorArmNavActorCritic,
)
from utils.experiment_utils import Builder


class ArmNavMixInResNetGRUConfig(ArmNavBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        if any(isinstance(s, RGBSensor) for s in cls.SENSORS):
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=["rgb_lowres"],
                    output_uuid="rgb_resnet",
                )
            )


        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorArmNavActorCritic(
            action_space=gym.spaces.Discrete(len(PickUpDropOffTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
        )
