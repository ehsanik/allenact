import random
import warnings
from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np

from plugins.ithor_arm_plugin.ithor_arm_environment import IThorMidLevelEnvironment
from utils.debugger_util import ForkedPdb
from plugins.ithor_arm_plugin.ithor_arm_constants import MOVE_ARM_CONSTANT
from plugins.ithor_arm_plugin.ithor_arm_viz import LoggerVisualizer

# DONE = "Done"
MOVE_AHEAD = "MoveAheadContinuous"
ROTATE_LEFT = "RotateLeftContinuous"
ROTATE_RIGHT = "RotateRightContinuous"
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_X_P = "MoveArmXP"
MOVE_ARM_X_M = "MoveArmXM"
MOVE_ARM_Y_P = "MoveArmYP"
MOVE_ARM_Y_M = "MoveArmYM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
PICKUP = 'PickUpMidLevel'
DONE = 'DoneMidLevel'
# MidPick = "PickUpMidLevelHand"

#KIANA later on add end action and hand rotations and drop action

from core.base_abstractions.misc import RLStepResult
from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import Task

def position_distance(s1, s2):
    position1 = s1['position']
    position2 = s2['position']
    return ((position1['x'] - position2['x'])**2 + (position1['y'] - position2['y'])**2 + (position1['z'] - position2['z'])**2) ** 0.5



class MidArmTask(Task[IThorMidLevelEnvironment]):

    _actions = ()

    def __init__(
            self,
            env: IThorMidLevelEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            visualizers: List[LoggerVisualizer] = [],
            **kwargs
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible: Optional[
            List[Tuple[float, float, int, int]]
        ] = None
        self.visualizers = visualizers
        self.start_visualize()


    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def obj_state_aproximity(self, s1, s2):
        #KIANA ignore rotation for now
        position1 = s1['position']
        position2 = s2['position']
        eps = MOVE_ARM_CONSTANT
        return (abs(position1['x'] - position2['x']) < eps and abs(position1['y'] - position2['y']) < eps and abs(position1['z'] - position2['z']) < eps)


    def start_visualize(self):
        for visualizer in self.visualizers:
            # assert visualizer.is_empty(), ForkedPdb().set_trace()
            #TODO this is a quick hack, fix it later, why finish visualizer is not called?
            if not visualizer.is_empty():
                visualizer.finish_episode(self, self.task_info)
                visualizer.finish_episode_metrics(self, self.task_info, None)
            # image = self.env.current_frame #Adding first frame
            visualizer.log(self.env, "")

    def visualize(self, action_str):

        for vizualizer in self.visualizers:
            vizualizer.log(self.env, action_str)

    def finish_visualizer(self, episode_success):

        for visualizer in self.visualizers:
            visualizer.finish_episode(self, self.task_info)

    def finish_visualizer_metrics(self, metric_results):

        for visualizer in self.visualizers:
            visualizer.finish_episode_metrics(self, self.task_info, metric_results)

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame



    def metrics(self) -> Dict[str, Any]:
        result = {}
        action_stat = {'metric/action_stat/' + action_str: 0 for action_str in self._actions}
        result = {**action_stat, **result}
        result['metric/action_stat/' + self._last_action_str] = 1
        last_action_name = self._last_action_str
        last_action_success = self.last_action_success
        result['metric/' + 'action_success/{}'.format(last_action_name)] = int(last_action_success)
        result['metric/' + 'action_success/total'] = int(last_action_success)

        self.visualize(self._last_action_str)

        if self.is_done():
            result = {**result, **super(MidArmTask, self).metrics()}
            self.finish_visualizer(self._success)
            if self._success:
                result['metric/average/eplen_success'] = result['ep_length']
            result['success'] = self._success

        return result


    def _step(self, action: int) -> RLStepResult:

        raise Exception('Not Implemented')


    def judge(self) -> float:
        raise Exception('Not Implemented')


class PickUpDropOffTask(MidArmTask):

    _actions = (MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M)# put back, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT)#LATER_TODO do the actual pickup later, PICKUP)

    def __init__(
            self,
            env: IThorMidLevelEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            visualizers: List[LoggerVisualizer] = [],
            **kwargs
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible: Optional[
            List[Tuple[float, float, int, int]]
        ] = None
        self.visualizers = visualizers

        #in allenact initialization is with 0.2
        self.last_obj_to_goal_distance = None
        self.last_arm_to_obj_distance = None
        self.object_picked_up = False
        self.got_reward_for_pickup = False
        self.start_visualize()



    def metrics(self) -> Dict[str, Any]:
        result = super(PickUpDropOffTask, self).metrics()
        if self.is_done():
            final_obj_distance_from_goal = self.obj_distance_from_goal()
            result['metric/average/final_obj_distance_from_goal'] = final_obj_distance_from_goal
            final_arm_distance_from_obj = self.arm_distance_from_obj()
            result['metric/average/final_arm_distance_from_obj'] = final_arm_distance_from_obj
            final_obj_pickup = 1 if self.object_picked_up else 0
            result['metric/average/final_obj_pickup'] = final_obj_pickup

            original_distance = self.get_original_object_distance()
            result['metric/average/original_distance'] = original_distance

            # this ratio can be more than 1?
            if self.object_picked_up:
                ratio_distance_left = final_obj_distance_from_goal / original_distance
                result['metric/average/ratio_distance_left'] = ratio_distance_left
                result['metric/average/eplen_pickup'] = self.eplen_pickup

            self.finish_visualizer_metrics(result)
        return result


    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self._last_action_str = action_str

        self.env.step({"action": action_str})


        self.last_action_success = self.env.last_action_success

        # just check whether the object is within the reach, if yes, pick up
        object_id = self.task_info['objectId']

        success_finished_task = False

        if not self.object_picked_up:

            pickupable_objects = self.env.get_pickupable_objects()

            if object_id in pickupable_objects:
                # self.env.finish_and_show_off() # remove this, it's just for visualization purposes
                success = self.env.pickup_object(self.task_info['objectId'])
                # make sure the result of above is true before setting the following to true
                if success:
                    self.object_picked_up = True
                    self.eplen_pickup = self._num_steps_taken + 1 # plus one because this step has not been counted yet

        else:
            object_state = self.env.get_object_by_id(object_id)
            goal_state = self.task_info['target_location']
            if self.object_picked_up and self.obj_state_aproximity(object_state, goal_state):
                success_finished_task = True

        if success_finished_task:
            self._took_end_action = True
            self._success = True
            self.last_action_success = True

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def arm_distance_from_obj(self):
        goal_obj_id = self.task_info['objectId']
        object_info = self.env.get_object_by_id(goal_obj_id)
        hand_state = self.env.get_absolute_hand_state()
        return position_distance(object_info, hand_state)

    def obj_distance_from_goal(self):
        goal_obj_id = self.task_info['objectId']
        object_info = self.env.get_object_by_id(goal_obj_id)
        goal_state = self.task_info['target_location']
        return position_distance(object_info, goal_state)

    def get_original_object_distance(self):
        goal_obj_id = self.task_info['objectId']
        s_init = dict(position=self.task_info['source_location']['object_location'])
        current_location = self.env.get_object_by_id(goal_obj_id)

        original_object_distance = position_distance(s_init, current_location)
        return original_object_distance




    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = -0.01

        if not self.last_action_success:
            reward += -0.03

        if self._took_end_action:
            reward += 10.0 if self._success else 0

        if self._last_action_str in [PICKUP, DONE]: #TODO this needs to be removed later, just a sanity check
            reward -= 5

        #increase reward if object pickup and only do it once
        if not self.got_reward_for_pickup and self.object_picked_up:
            reward += 5
            self.got_reward_for_pickup = True

        current_obj_to_arm_distance = self.arm_distance_from_obj()
        if self.last_arm_to_obj_distance is None:
            delta_arm_to_obj_distance_reward = 0
        else:
            delta_arm_to_obj_distance_reward = self.last_arm_to_obj_distance - current_obj_to_arm_distance
        self.last_arm_to_obj_distance = current_obj_to_arm_distance
        reward += delta_arm_to_obj_distance_reward

        current_obj_to_goal_distance = self.obj_distance_from_goal()
        if self.last_obj_to_goal_distance is None:
            delta_obj_to_goal_distance_reward = 0
        else:
            delta_obj_to_goal_distance_reward = self.last_obj_to_goal_distance - current_obj_to_goal_distance
        self.last_obj_to_goal_distance = current_obj_to_goal_distance
        reward += delta_obj_to_goal_distance_reward


        # distance * 0.1 does not make sense because then it will not take any actions

        # add collision cost, maybe distance to goal objective,...

        return float(reward)


class PickUpDropOffTaskWNav(PickUpDropOffTask):
    _actions = (MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, PICKUP, DONE)