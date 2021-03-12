import ai2thor
import ai2thor.fifo_server

# BUILD_NUMBER = None
# ARM_BUILD_NUMBER = 'a2ff21ad7fa8409b188dbd30781e448a4cf2c8fe'
# ARM_BUILD_NUMBER = '2afa985d898e961db57de3e3d582a366e0cc7a41'
# ARM_BUILD_NUMBER = '65cb9da687c30a8a738e00b8bb21d7536b83fd07'
# ARM_BUILD_NUMBER = 'df6c729530c0ce9d41f98b5c59f0293ecccaad42'
# ARM_BUILD_NUMBER = '51aacd8a06e44412afe6ce9e046f05a481e16939'
# ARM_BUILD_NUMBER = '2a98a22cfd2e59b2dbdd7bfa036f191afd305fad'
# ARM_BUILD_NUMBER = '12a36839a9670e70ffe6f2171212147ce306d818'
# ARM_BUILD_NUMBER = 'a1d3d6ad89b7ec06b3a406f391e847222dde5a37'
# ARM_BUILD_NUMBER = '52f8df9cc9fc23dc8a4387fb29d7fd2cbdb22d53'
from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from utils.debugger_util import ForkedPdb

ARM_BUILD_NUMBER = '6da9163ea2b6766a632d27bc14ebacee7b9cf9fa'

ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
MOVE_ARM_CONSTANT = 0.05 #Do we need to change this?
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT # Can not do the next one because the changes in height becomes random. maybe look into this later on
# MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT / (ARM_MAX_HEIGHT - ARM_MIN_HEIGHT) #LATER_TODO

ADITIONAL_ARM_ARGS = {
    'disableRendering': True,
    'restrictMovement': False,
    'waitForFixedUpdate': False,
    'returnToStart': True,
    'speed': 1,
    'moveSpeed': 1,
    # 'move_constant': 0.05,
}

ENV_ARGS = dict(
    gridSize=0.25,
    width=224, height=224,
    visibilityDistance=1.0,
    agentMode='arm', fieldOfView=100,
    agentControllerType='mid-level',
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold = True, massThreshold = 10,
    autoSimulation=False, autoSyncTransforms=True
    )

TRAIN_OBJECTS = ['Apple', 'Bread', 'Tomato', 'Lettuce', 'Pot', 'Mug']
TEST_OBJECTS = ['Potato', 'SoapBottle', 'Pan', 'Egg', 'Spatula', 'Cup']


def make_all_objects_unbreakable(controller):
        all_breakable_objects = [o['objectType'] for o in controller.last_event.metadata['objects'] if o['breakable'] is True]
        all_breakable_objects = set(all_breakable_objects)
        for obj_type in all_breakable_objects:
            controller.step(action='MakeObjectsOfTypeUnbreakable', objectType=obj_type)


def reset_environment_and_additional_commands(controller, scene_name):
    controller.reset(scene_name)
    controller.step('PausePhysicsAutoSim', autoSyncTransforms=False)
    controller.step(action='MakeAllObjectsMoveable')
    make_all_objects_unbreakable(controller)
    return

# Apple, Bread, Tomato, Lettuce, Pot, Mug,Potato, SoapBottle, Pan, Egg, Spatula, Cup

def transport_wrapper(controller, target_object, target_location):
    action_detail_list = []
    transport_detail = dict(action='PlaceObjectAtPoint', objectId=target_object, position=target_location, forceKinematic=True)
    action_detail_list.append(transport_detail)
    # controller.step('PhysicsSyncTransforms')
    advance_detail = dict(action='AdvancePhysicsStep', simSeconds=1.0)
    action_detail_list.append(advance_detail)

    if issubclass(type(controller), IThorEnvironment):
        event = controller.step(transport_detail)
        controller.step(advance_detail)
    elif type(controller) == ai2thor.controller.Controller:
        event = controller.step(**transport_detail)
        controller.step(**advance_detail)
    return event


VALID_OBJECT_LIST = ['Knife', 'Bread', 'Fork', 'Potato', 'SoapBottle', 'Pan', 'Plate', 'Tomato', 'Egg', 'Pot', 'Spatula', 'Cup', 'Bowl', 'SaltShaker', 'PepperShaker', 'Lettuce', 'ButterKnife', 'Apple', 'DishSponge', 'Spoon', 'Mug']

import json
with open('datasets/ithor-armnav/ideal_pose.json') as f:
    scene_start_cheating_init_pose = json.load(f)
# scene_start_cheating_init_pose = {"FloorPlan1_physics": {"x": -1.0, "y": 0.9009995460510254, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan2_physics": {"x": -1.0, "y": 0.9009992480278015, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan3_physics": {"x": 0.0, "y": 1.1232060194015503, "z": -1.75, "rotation": 0, "horizon": 10}, "FloorPlan4_physics": {"x": -2.0, "y": 0.900999903678894, "z": 1.25, "rotation": 0, "horizon": 10}, "FloorPlan5_physics": {"x": 0.75, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan6_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan7_physics": {"x": 1.25, "y": 0.9009991884231567, "z": -0.5, "rotation": 0, "horizon": 10}, "FloorPlan8_physics": {"x": -1.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan9_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan10_physics": {"x": 0.0, "y": 0.9009992480278015, "z": -1.25, "rotation": 0, "horizon": 10}, "FloorPlan11_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan12_physics": {"x": 0.5, "y": 0.9799999594688416, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan13_physics": {"x": -2.0, "y": 0.8995019197463989, "z": 3.75, "rotation": 0, "horizon": 10}, "FloorPlan14_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 90, "horizon": 10}, "FloorPlan15_physics": {"x": -1.5, "y": 0.914953351020813, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan16_physics": {"x": 1.25, "y": 0.9037266969680786, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan17_physics": {"x": 0.0, "y": 0.9089995622634888, "z": 1.75, "rotation": 0, "horizon": 10}, "FloorPlan18_physics": {"x": -0.5, "y": 0.9009998440742493, "z": 2.25, "rotation": 0, "horizon": 10}, "FloorPlan19_physics": {"x": -1.25, "y": 0.9023619890213013, "z": -2.0, "rotation": 0, "horizon": 10}, "FloorPlan20_physics": {"x": 1.0, "y": 0.9009991884231567, "z": -1.0, "rotation": 0, "horizon": 10}, "FloorPlan21_physics": {"x": 0.0, "y": 0.8696962594985962, "z": -2.75, "rotation": 0, "horizon": 10}, "FloorPlan22_physics": {"x": -1.5, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan23_physics": {"x": -3.0, "y": 0.9009994864463806, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan24_physics": {"x": -0.5, "y": 0.9009992480278015, "z": 2.5, "rotation": 0, "horizon": 10}, "FloorPlan25_physics": {"x": -2.0, "y": 0.9009992480278015, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan26_physics": {"x": -1.0, "y": 0.9015910625457764, "z": 1.5, "rotation": 0, "horizon": 10}, "FloorPlan27_physics": {"x": 1.0, "y": 0.9010001420974731, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan28_physics": {"x": -1.5, "y": 0.9009982347488403, "z": -1.5, "rotation": 0, "horizon": 10}, "FloorPlan29_physics": {"x": 1.25, "y": 0.9317981004714966, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan30_physics": {"x": 0.25, "y": 0.9277887344360352, "z": -0.5, "rotation": 0, "horizon": 10}}


# side_arm_start_cheating_init_pose = {"FloorPlan1_physics": {"x": -1.0, "y": 0.9009995460510254, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan2_physics": {"x": -1.0, "y": 0.9009992480278015, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan3_physics": {"x": 0.0, "y": 1.1232060194015503, "z": -1.75, "rotation": 180, "horizon": 10}, "FloorPlan4_physics": {"x": -2.0, "y": 0.900999903678894, "z": 1.25, "rotation": 0, "horizon": 10}, "FloorPlan5_physics": {"x": 0.75, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan6_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan7_physics": {"x": 1.25, "y": 0.9009991884231567, "z": -0.5, "rotation": 0, "horizon": 10}, "FloorPlan8_physics": {"x": -1.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan9_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan10_physics": {"x": 0.0, "y": 0.9009992480278015, "z": -1.25, "rotation": 0, "horizon": 10}, "FloorPlan11_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan12_physics": {"x": 0.5, "y": 0.9799999594688416, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan13_physics": {"x": -2.0, "y": 0.8995019197463989, "z": 3.75, "rotation": 0, "horizon": 10}, "FloorPlan14_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 90, "horizon": 10}, "FloorPlan15_physics": {"x": -1.5, "y": 0.914953351020813, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan16_physics": {"x": 1.25, "y": 0.9037266969680786, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan17_physics": {"x": 0.0, "y": 0.9089995622634888, "z": 1.75, "rotation": 0, "horizon": 10}, "FloorPlan18_physics": {"x": -0.5, "y": 0.9009998440742493, "z": 2.25, "rotation": 0, "horizon": 10}, "FloorPlan19_physics": {"x": -1.25, "y": 0.9023619890213013, "z": -2.0, "rotation": 0, "horizon": 10}, "FloorPlan20_physics": {"x": 1.0, "y": 0.9009991884231567, "z": -1.0, "rotation": 180, "horizon": 10}, "FloorPlan21_physics": {"x": 0.0, "y": 0.8696962594985962, "z": -2.75, "rotation": 90, "horizon": 10}, "FloorPlan22_physics": {"x": -1.5, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan23_physics": {"x": -3.0, "y": 0.9009994864463806, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan24_physics": {"x": -0.5, "y": 0.9009992480278015, "z": 2.5, "rotation": 0, "horizon": 10}, "FloorPlan25_physics": {"x": -2.0, "y": 0.9009992480278015, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan26_physics": {"x": -1.0, "y": 0.9015910625457764, "z": 1.5, "rotation": 0, "horizon": 10}, "FloorPlan27_physics": {"x": 1.0, "y": 0.9010001420974731, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan28_physics": {"x": -1.5, "y": 0.9009982347488403, "z": -1.5, "rotation": 0, "horizon": 10}, "FloorPlan29_physics": {"x": 1.25, "y": 0.9317981004714966, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan30_physics": {"x": 0.25, "y": 0.9277887344360352, "z": -0.5, "rotation": 90, "horizon": 10}}