import copy
import json
import random
from typing import List, Dict, Optional, Any, Union

import cv2
import gym

from core.base_abstractions.task import Task

from plugins.ithor_arm_plugin.ithor_arm_tasks import PickUpDropOffTask
from plugins.ithor_arm_plugin.ithor_arm_environment import IThorMidLevelEnvironment, IThorMidLevelDepthEnvironment
from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import TaskSampler
from utils.debugger_util import ForkedPdb
from utils.experiment_utils import set_deterministic_cudnn, set_seed
from plugins.ithor_arm_plugin.ithor_arm_constants import scene_start_cheating_init_pose, ADITIONAL_ARM_ARGS
from utils.system import get_logger
from plugins.ithor_arm_plugin.ithor_arm_viz import LoggerVisualizer, TestMetricLogger


class MidLevelArmTaskSampler(TaskSampler):

    _TASK_TYPE = Task


    def __init__(
            self,
            scenes: List[str],
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            objects: List[str],
            scene_period: Optional[Union[int, str]] = None,
            max_tasks: Optional[int] = None,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            fixed_tasks: Optional[List[Dict[str, Any]]] = None,
            visualizers: List[LoggerVisualizer] = [],
            *args,
            **kwargs
    ) -> None:

        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[IThorMidLevelEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.objects = objects

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[Task] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()
        self.visualizers = visualizers
        self.sampler_mode = kwargs['sampler_mode']
        self.cap_training = kwargs['cap_training']


        if self.sampler_mode == 'test':
            self.visualizers.append(TestMetricLogger(exp_name=kwargs['exp_name']))


    def _create_environment(self, **kwargs) -> IThorMidLevelEnvironment:
        env = IThorMidLevelEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            # restrict_to_initially_reachable_points=True, # is this really important?
            env_args=self.env_args, #TODO why can't i do the same thing as the other task sampler?
        )

        return env

    # # remove these two
    # @property
    # def length(self) -> Union[int, float]:
    #     """Length.
    #
    #     # Returns
    #
    #     Number of total tasks remaining that can be sampled. Can be float('inf').
    #     """
    #     return float("inf") if self.max_tasks is None else self.max_tasks
    #
    # @property
    # def total_unique(self) -> Optional[Union[int, float]]:
    #     # raise Exception('need to define this')
    #     return None

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    # def sample_scene(self, force_advance_scene: bool): Removed this because I can handle this

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.sampler_index = 0

        self.max_tasks = self.reset_tasks


    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)



class PickupDropOffGeneralSampler(MidLevelArmTaskSampler):

    _TASK_TYPE = PickUpDropOffTask

    def __init__(
            self,
            **kwargs
    ) -> None:

        super(PickupDropOffGeneralSampler, self).__init__(**kwargs)
        #LATER_TODO change this later
        self.all_possible_points = []


        for scene in self.scenes:
            for object in self.objects:
                valid_position_adr = 'datasets/ithor-armnav/valid_{}_positions_in_{}.json'.format(object, scene)
                try:
                    with open(valid_position_adr) as f:
                        data_points = json.load(f)
                except Exception:
                    print('Failed to load', valid_position_adr)
                    continue
                visible_data = [data for data in data_points[scene] if data['visibility']]
                self.all_possible_points += visible_data

        self.countertop_object_to_data_id = self.calc_possible_trajectories(self.all_possible_points)

        scene_names = set([self.all_possible_points[counter[0]]['scene_name'] for counter in self.countertop_object_to_data_id.values() if len(counter) > 1])

        if len(set(scene_names)) < len(self.scenes):
            print('Not all scenes appear')


        if self.cap_training is not None:
            # To be consistent across runs
            import random
            random.seed(0)
            for countertop_obj in self.countertop_object_to_data_id.keys():
                all_sequence = self.countertop_object_to_data_id[countertop_obj]
                count_to_keep = int(len(all_sequence) * self.cap_training)
                if count_to_keep <= 2:
                    count_to_keep = len(all_sequence)
                all_sequence = random.sample(all_sequence, count_to_keep)
                self.countertop_object_to_data_id[countertop_obj] = all_sequence



        print('Len dataset', len(self.all_possible_points), 'total_remained', sum([len(v) for v in self.countertop_object_to_data_id.values()]))

        if self.sampler_mode != 'train': # Be aware that this totally overrides some stuff
            self.deterministic_data_list = []
            for scene in self.scenes:
                for object in self.objects:
                    valid_position_adr = 'datasets/ithor-armnav/w_nav_tasks_{}_positions_in_{}.json'.format(object, scene)
                    try:
                        with open(valid_position_adr) as f:
                            data_points = json.load(f)
                    except Exception:
                        print('Failed to load', valid_position_adr)
                        continue
                    visible_data = [data for data in data_points[scene]]
                    self.deterministic_data_list += visible_data

                    # [v[0]['countertop_id'] for v in visible_data]

        if self.sampler_mode == 'test':
            random.shuffle(self.deterministic_data_list)
            # very patched up
            self.max_tasks = self.reset_tasks = len(self.deterministic_data_list)





    def next_task(self, force_advance_scene: bool = False) -> Optional[PickUpDropOffTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != 'train' and self.length <= 0: #TODO I added this but why?
            return None


        source_data_point, target_data_point = self.get_source_target_indices()

        scene = source_data_point['scene_name']

        assert source_data_point['object_id'] == target_data_point['object_id']
        assert source_data_point['scene_name'] == target_data_point['scene_name']



        if self.env is None:
            self.env = self._create_environment()

        self.env.reset(scene_name=scene, agentMode="arm", agentControllerType="mid-level")

        source_location = source_data_point
        target_location = dict(position=target_data_point['object_location'], rotation = {'x':0, 'y':0, 'z':0})


        task_info = {
            'objectId': source_location['object_id'],
            'countertop_id': source_location['countertop_id'],
            "source_location": source_location,
            'target_location': target_location,
        }

        event = self.env.controller.step(action = 'PlaceObjectAtPoint', objectId=source_location['object_id'], position=source_location['object_location'])
        if event.metadata['lastActionSuccess'] == False:
            print('oh no could not transport')
        agent_state = source_location['agent_pose']


        # for start arm from high up as a cheating, this block is very important. never remove
        initial_pose = scene_start_cheating_init_pose[scene]
        event1 = self.env.controller.step(action='TeleportFull', x=initial_pose['x'], y=initial_pose['y'], z=initial_pose['z'], rotation=dict(x=0, y=initial_pose['rotation'], z=0), horizon=initial_pose['horizon'])
        self.env.controller.step('PausePhysicsAutoSim')
        event2 = self.env.controller.step(action='MoveMidLevelArm',  position=dict(x=0.0, y=0, z=0.35), **ADITIONAL_ARM_ARGS)
        event3 = self.env.controller.step(action='MoveMidLevelArmHeight', y=0.8, **ADITIONAL_ARM_ARGS)
        if not(event1.metadata['lastActionSuccess'] and event2.metadata['lastActionSuccess'] and event3.metadata['lastActionSuccess']):
            print('ARM MOVEMENT FAILED> SHOUD NEVER HAPPEN')
            # print('scene', scene, initial_pose, ADITIONAL_ARM_ARGS)
            # print(event1.metadata['actionReturn'] , event2.metadata['actionReturn'] , event3.metadata['actionReturn'])


        event = self.env.controller.step(action='TeleportFull', x=agent_state['position']['x'], y=agent_state['position']['y'], z=agent_state['position']['z'], rotation=dict(x=agent_state['rotation']['x'], y=agent_state['rotation']['y'], z=agent_state['rotation']['z']), horizon=agent_state['cameraHorizon'])
        if event.metadata['lastActionSuccess'] == False:
            print('oh no could not teleport')

        # remove this
        if self.env._verbose:
            print('task: ', task_info['objectId'], task_info['countertop_id'], 'source_location', task_info['source_location'], 'target_location', task_info['target_location'], )
            print('counter_top_source', source_data_point['countertop_id'], 'counter_top_target', target_data_point['countertop_id'])

        self._last_sampled_task = self._TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
        )
        return self._last_sampled_task


    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        if self.sampler_mode == 'train':
            return None
        else:
            return min(self.max_tasks, len(self.deterministic_data_list))

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self.total_unique - self.sampler_index if self.sampler_mode != 'train' else (float("inf") if self.max_tasks is None else self.max_tasks)




    def get_source_target_indices(self):
        if self.sampler_mode == 'train':
            valid_countertops = [k for (k, v) in self.countertop_object_to_data_id.items() if len(v) > 1]
            countertop_id = random.choice(valid_countertops)
            indices = random.sample(self.countertop_object_to_data_id[countertop_id], 2)
            result = self.all_possible_points[indices[0]],self.all_possible_points[indices[1]]
        else:
            result = self.deterministic_data_list[self.sampler_index]
            # ForkedPdb().set_trace()
            self.sampler_index += 1
            # self.max_tasks -= 1

        return result


    def calc_possible_trajectories(self, all_possible_points):

        object_to_data_id = {}

        for i in range(len(all_possible_points)):
            # countertop_id = all_possible_points[i]['countertop_id']
            object_id = all_possible_points[i]['object_id']
            # counter_object = '{}_{}'.format(countertop_id, object_id)
            # countertop_object_to_data_id.setdefault(counter_object, [])
            # countertop_object_to_data_id[counter_object].append(i)
            #
            object_to_data_id.setdefault(object_id, [])
            object_to_data_id[object_id].append(i)

        return object_to_data_id

    # def generate_all_possible_sequences(self):
    #     result = []
    #     for (k, v) in self.countertop_object_to_data_id.items():
    #         if len(v) <= 1:
    #             continue
    #         result += get_all_tuples_from_list(v)
    #
    #     return result


class DepthPickDropGeneralSampler(PickupDropOffGeneralSampler):

    def _create_environment(self) -> IThorMidLevelEnvironment:
        env = IThorMidLevelDepthEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            # restrict_to_initially_reachable_points=True, # is this really important?
            **self.env_args,
        )

        return env

class PickupDropOffGeneralSamplerDeterministic(PickupDropOffGeneralSampler): #TODO all the few shots should be with this one

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if True: # Be aware that this totally overrides some stuff
            self.deterministic_data_list = []
            for scene in self.scenes:
                for object in self.objects:
                    valid_position_adr = 'datasets/ithor-armnav/w_nav_tasks_{}_positions_in_{}.json'.format(object, scene)
                    try:
                        with open(valid_position_adr) as f:
                            data_points = json.load(f)
                    except Exception:
                        print('Failed to load', valid_position_adr)
                        continue
                    visible_data = [data for data in data_points[scene]]
                    self.deterministic_data_list += visible_data

                    # [v[0]['countertop_id'] for v in visible_data]
            random.shuffle(self.deterministic_data_list)
            self.sampler_index = 0
            self.epoch = 0
        if self.sampler_mode == 'test':
            # very patched up
            self.max_tasks = self.reset_tasks = len(self.deterministic_data_list)


    def get_source_target_indices(self):
        if self.sampler_mode == 'train':
            if self.sampler_index >= len(self.deterministic_data_list):
                random.shuffle(self.deterministic_data_list)
                self.epoch += 1
                print('Just finished epoch ', self.epoch)
            self.sampler_index %= len(self.deterministic_data_list)
            # valid_countertops = [k for (k, v) in self.countertop_object_to_data_id.items() if len(v) > 1]
            # countertop_id = random.choice(valid_countertops)
            # result = random.sample(self.countertop_object_to_data_id[countertop_id], 2)

        result = self.deterministic_data_list[self.sampler_index]
        self.sampler_index += 1
        # self.max_tasks -= 1

        return result


def get_all_tuples_from_list(list):
    result = []
    for first_ind in range(len(list) - 1):
        for second_ind in range(first_ind + 1, len(list)):
            result.append([list[first_ind], list[second_ind]])
    return result

