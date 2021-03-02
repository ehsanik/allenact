# ADITIONAL_ARM_ARGS = {
#     'disableRendering': False,
#     'restrictMovement': False,
#     'waitForFixedUpdate': False,
#     'returnToStart': True,
#     'speed': 2,
#     'moveSpeed': 5,
# }

# BUILD_NUMBER = None
# ARM_BUILD_NUMBER = 'a2ff21ad7fa8409b188dbd30781e448a4cf2c8fe'
# ARM_BUILD_NUMBER = '2afa985d898e961db57de3e3d582a366e0cc7a41'
# ARM_BUILD_NUMBER = '65cb9da687c30a8a738e00b8bb21d7536b83fd07'
# ARM_BUILD_NUMBER = 'df6c729530c0ce9d41f98b5c59f0293ecccaad42'
# ARM_BUILD_NUMBER = '51aacd8a06e44412afe6ce9e046f05a481e16939'
ARM_BUILD_NUMBER = '90e99f80f21889ace9afeed94b5cbca94facd3e7' #TODO center arm

ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
MOVE_ARM_CONSTANT = 0.05 #Do we need to change this?
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT # Can not do the next one because the changes in height becomes random. maybe look into this later on
# MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT / (ARM_MAX_HEIGHT - ARM_MIN_HEIGHT) #TODO this is very important

ADITIONAL_ARM_ARGS = {
    'disableRendering': True,
    'restrictMovement': False,
    'waitForFixedUpdate': False,
    'returnToStart': True,
    'speed': 1,
    'moveSpeed': 1,
    # 'move_constant': 0.05,
}

TRAIN_OBJECTS = ['Apple', 'Bread', 'Tomato', 'Lettuce', 'Pot', 'Mug']
TEST_OBJECTS = ['Potato', 'SoapBottle', 'Pan', 'Egg', 'Spatula', 'Cup']



# Apple, Bread, Tomato, Lettuce, Pot, Mug,Potato, SoapBottle, Pan, Egg, Spatula, Cup



VALID_OBJECT_LIST = ['Knife', 'Bread', 'Fork', 'Potato', 'SoapBottle', 'Pan', 'Plate', 'Tomato', 'Egg', 'Pot', 'Spatula', 'Cup', 'Bowl', 'SaltShaker', 'PepperShaker', 'Lettuce', 'ButterKnife', 'Apple', 'DishSponge', 'Spoon', 'Mug']

import json
with open('datasets/ithor-armnav/ideal_pose.json') as f:
    scene_start_cheating_init_pose = json.load(f)
# scene_start_cheating_init_pose = {"FloorPlan1_physics": {"x": -1.0, "y": 0.9009995460510254, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan2_physics": {"x": -1.0, "y": 0.9009992480278015, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan3_physics": {"x": 0.0, "y": 1.1232060194015503, "z": -1.75, "rotation": 0, "horizon": 10}, "FloorPlan4_physics": {"x": -2.0, "y": 0.900999903678894, "z": 1.25, "rotation": 0, "horizon": 10}, "FloorPlan5_physics": {"x": 0.75, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan6_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan7_physics": {"x": 1.25, "y": 0.9009991884231567, "z": -0.5, "rotation": 0, "horizon": 10}, "FloorPlan8_physics": {"x": -1.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan9_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan10_physics": {"x": 0.0, "y": 0.9009992480278015, "z": -1.25, "rotation": 0, "horizon": 10}, "FloorPlan11_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan12_physics": {"x": 0.5, "y": 0.9799999594688416, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan13_physics": {"x": -2.0, "y": 0.8995019197463989, "z": 3.75, "rotation": 0, "horizon": 10}, "FloorPlan14_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 90, "horizon": 10}, "FloorPlan15_physics": {"x": -1.5, "y": 0.914953351020813, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan16_physics": {"x": 1.25, "y": 0.9037266969680786, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan17_physics": {"x": 0.0, "y": 0.9089995622634888, "z": 1.75, "rotation": 0, "horizon": 10}, "FloorPlan18_physics": {"x": -0.5, "y": 0.9009998440742493, "z": 2.25, "rotation": 0, "horizon": 10}, "FloorPlan19_physics": {"x": -1.25, "y": 0.9023619890213013, "z": -2.0, "rotation": 0, "horizon": 10}, "FloorPlan20_physics": {"x": 1.0, "y": 0.9009991884231567, "z": -1.0, "rotation": 0, "horizon": 10}, "FloorPlan21_physics": {"x": 0.0, "y": 0.8696962594985962, "z": -2.75, "rotation": 0, "horizon": 10}, "FloorPlan22_physics": {"x": -1.5, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan23_physics": {"x": -3.0, "y": 0.9009994864463806, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan24_physics": {"x": -0.5, "y": 0.9009992480278015, "z": 2.5, "rotation": 0, "horizon": 10}, "FloorPlan25_physics": {"x": -2.0, "y": 0.9009992480278015, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan26_physics": {"x": -1.0, "y": 0.9015910625457764, "z": 1.5, "rotation": 0, "horizon": 10}, "FloorPlan27_physics": {"x": 1.0, "y": 0.9010001420974731, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan28_physics": {"x": -1.5, "y": 0.9009982347488403, "z": -1.5, "rotation": 0, "horizon": 10}, "FloorPlan29_physics": {"x": 1.25, "y": 0.9317981004714966, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan30_physics": {"x": 0.25, "y": 0.9277887344360352, "z": -0.5, "rotation": 0, "horizon": 10}}


# side_arm_start_cheating_init_pose = {"FloorPlan1_physics": {"x": -1.0, "y": 0.9009995460510254, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan2_physics": {"x": -1.0, "y": 0.9009992480278015, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan3_physics": {"x": 0.0, "y": 1.1232060194015503, "z": -1.75, "rotation": 180, "horizon": 10}, "FloorPlan4_physics": {"x": -2.0, "y": 0.900999903678894, "z": 1.25, "rotation": 0, "horizon": 10}, "FloorPlan5_physics": {"x": 0.75, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan6_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan7_physics": {"x": 1.25, "y": 0.9009991884231567, "z": -0.5, "rotation": 0, "horizon": 10}, "FloorPlan8_physics": {"x": -1.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan9_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan10_physics": {"x": 0.0, "y": 0.9009992480278015, "z": -1.25, "rotation": 0, "horizon": 10}, "FloorPlan11_physics": {"x": 0.0, "y": 0.9009991884231567, "z": -0.75, "rotation": 0, "horizon": 10}, "FloorPlan12_physics": {"x": 0.5, "y": 0.9799999594688416, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan13_physics": {"x": -2.0, "y": 0.8995019197463989, "z": 3.75, "rotation": 0, "horizon": 10}, "FloorPlan14_physics": {"x": 0.0, "y": 0.9009991884231567, "z": 0.0, "rotation": 90, "horizon": 10}, "FloorPlan15_physics": {"x": -1.5, "y": 0.914953351020813, "z": 2.0, "rotation": 0, "horizon": 10}, "FloorPlan16_physics": {"x": 1.25, "y": 0.9037266969680786, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan17_physics": {"x": 0.0, "y": 0.9089995622634888, "z": 1.75, "rotation": 0, "horizon": 10}, "FloorPlan18_physics": {"x": -0.5, "y": 0.9009998440742493, "z": 2.25, "rotation": 0, "horizon": 10}, "FloorPlan19_physics": {"x": -1.25, "y": 0.9023619890213013, "z": -2.0, "rotation": 0, "horizon": 10}, "FloorPlan20_physics": {"x": 1.0, "y": 0.9009991884231567, "z": -1.0, "rotation": 180, "horizon": 10}, "FloorPlan21_physics": {"x": 0.0, "y": 0.8696962594985962, "z": -2.75, "rotation": 90, "horizon": 10}, "FloorPlan22_physics": {"x": -1.5, "y": 0.9009991884231567, "z": 0.0, "rotation": 0, "horizon": 10}, "FloorPlan23_physics": {"x": -3.0, "y": 0.9009994864463806, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan24_physics": {"x": -0.5, "y": 0.9009992480278015, "z": 2.5, "rotation": 0, "horizon": 10}, "FloorPlan25_physics": {"x": -2.0, "y": 0.9009992480278015, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan26_physics": {"x": -1.0, "y": 0.9015910625457764, "z": 1.5, "rotation": 0, "horizon": 10}, "FloorPlan27_physics": {"x": 1.0, "y": 0.9010001420974731, "z": 1.0, "rotation": 0, "horizon": 10}, "FloorPlan28_physics": {"x": -1.5, "y": 0.9009982347488403, "z": -1.5, "rotation": 0, "horizon": 10}, "FloorPlan29_physics": {"x": 1.25, "y": 0.9317981004714966, "z": -0.5, "rotation": 90, "horizon": 10}, "FloorPlan30_physics": {"x": 0.25, "y": 0.9277887344360352, "z": -0.5, "rotation": 90, "horizon": 10}}
