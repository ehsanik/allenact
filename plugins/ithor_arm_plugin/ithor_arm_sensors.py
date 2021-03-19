import gym
import torch

from core.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from plugins.ithor_arm_plugin.arm_calculation_utils import convert_world_to_agent_coordinate, convert_state_to_tensor, diff_position
from plugins.ithor_arm_plugin.ithor_arm_constants import VALID_OBJECT_LIST
from plugins.ithor_arm_plugin.ithor_arm_environment import IThorMidLevelEnvironment
from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from core.base_abstractions.task import Task
import numpy as np
from typing import Dict, Any, List, Optional, Union

from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from utils.debugger_util import ForkedPdb
from utils.misc_utils import prepare_locals_for_super

class DepthSensorThor(
    DepthSensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment) -> np.ndarray:
        return env.controller.last_event.depth_frame.copy()

class BlindSensorThor(
    RGBSensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment) -> np.ndarray:
        result = env.current_frame.copy()
        result.fill(0)
        return result


class AgentRelativeCurrentObjectStateThorSensor(Sensor):
    def __init__(
            self,
            uuid: str = "relative_current_obj_state",
            **kwargs: Any
    ):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(low=-100,high=100, shape=(6,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self,
            env: IThorEnvironment,
            task: Task,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        object_id = task.task_info['objectId']
        current_object_state = env.get_object_by_id(object_id)
        relative_current_obj = convert_world_to_agent_coordinate(current_object_state, env.controller.last_event.metadata['agent'])
        result = convert_state_to_tensor(dict(position=relative_current_obj['position'], rotation=relative_current_obj['rotation']))
        return result

# class AgentRelativeGoalObjectStateThorSensor(Sensor):
#     def __init__(
#             self,
#             uuid: str = "relative_goal_obj_state",
#             **kwargs: Any
#     ):
#         # observation_space = gym.spaces.Discrete(len(self.detector_types))
#         observation_space = gym.spaces.Box(low=-100,high=100, shape=(6,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#         print('Deprecated?')
#
#     def get_observation(
#             self,
#             env: IThorEnvironment,
#             task: Task,
#             *args: Any,
#             **kwargs: Any
#     ) -> Any:
#         goal_object_state = task.task_info['world_goal_obj_state']
#         relative_goal_obj = convert_world_to_agent_coordinate(goal_object_state, env.controller.last_event.metadata['agent'])
#         result = convert_state_to_tensor(dict(position=relative_goal_obj['position'], rotation=relative_goal_obj['rotation']))
#         # if torch.any(result != result) or torch.any(torch.isinf(result: #LATER_TODO remove this
#         #     ForkedPdb().set_trace()
#         return result

class RelativeObjectToGoalSensor(Sensor):
    def __init__(
            self,
            uuid: str = "relative_obj_to_goal",
            **kwargs: Any
    ):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(low=-100,high=100, shape=(3,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        # #KIANA not sure about low and high TODO observation space is total bullshit
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorMidLevelEnvironment,
            task: Task,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info['objectId']
        object_info = env.get_object_by_id(goal_obj_id)
        target_state = task.task_info['target_location']

        agent_state = env.controller.last_event.metadata['agent']

        relative_current_obj = convert_world_to_agent_coordinate(object_info, agent_state)
        relative_goal_state = convert_world_to_agent_coordinate(target_state, agent_state)
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = convert_state_to_tensor(dict(position=relative_distance))
        # if torch.any(result != result) or torch.any(torch.isinf(result)): # remove this
        #     print('relative agent obj to goal')
        #     print(env.controller.last_event.metadata['agent']['position'])
        #     print(goal_obj_id)
        #     print('target state', target_state)
        #     result[:] = 0
        #     print('new result', result)
        return result


class RelativeAgentArmToObjectSensor(Sensor):# TODO double check this to see if it makes sense
    def __init__(
            self,
            uuid: str = "relative_agent_arm_to_obj",
            **kwargs: Any
    ):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(low=-100,high=100, shape=(3,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        # #KIANA not sure about low and high TODO observation space is total bullshit
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorMidLevelEnvironment,
            task: Task,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info['objectId']
        object_info = env.get_object_by_id(goal_obj_id)
        hand_state = env.get_absolute_hand_state()

        relative_goal_obj = convert_world_to_agent_coordinate(object_info, env.controller.last_event.metadata['agent'])
        relative_hand_state = convert_world_to_agent_coordinate(hand_state, env.controller.last_event.metadata['agent'])
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))
        # if torch.any(result != result) or torch.any(torch.isinf(result)): TODO remove this
        #     print('relative agent arm to obj')
        #     print(env.controller.last_event.metadata['agent']['position'])
        #     print('goal_obj_id', goal_obj_id)
        #     print('hand_state', hand_state)
        #     result[:] = 0
        #     print('new result', result)
        return result


class PickedUpObjSensor(Sensor):
    def __init__(
            self,
            uuid: str = "pickedup_object",
            **kwargs: Any
    ):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(low=-100,high=100, shape=(3,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        # #KIANA not sure about low and high TODO observation space is total bullshit
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorMidLevelEnvironment,
            task: Task,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        return task.object_picked_up

class ObjectTypeSensor(Sensor):
    def __init__(
            self,
            uuid: str = "object_type",
            **kwargs: Any
    ):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(low=-100,high=100, shape=(3,), dtype=np.float32)#(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        # #KIANA not sure about low and high TODO observation space is total bullshit
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorMidLevelEnvironment,
            task: Task,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        object_type = task.task_info['objectId'].split('|')[0]
        assert object_type in VALID_OBJECT_LIST
        index = VALID_OBJECT_LIST.index(object_type)
        one_hot = torch.zeros(len(VALID_OBJECT_LIST))
        one_hot[index] = 1
        return one_hot
