import glob
import os
import sys
import pdb
import cv2

BASE_DIR = '/Users/kianae/Desktop/visualization_cameras'
first_folder = 'first_person'
third_folder = 'third_party_camera'
combined_folder = 'combined'
video_folder = 'video'

first_folder, third_folder, combined_folder, video_folder = [os.path.join(BASE_DIR, x) for x in [first_folder, third_folder, combined_folder, video_folder]]

os.makedirs(combined_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

specific_video = None
RATE = 5

# specific_video = 32
# RATE = 25

# specific_video = 6
# RATE = 10
#
#
# specific_video = -1
# RATE = 1

if specific_video:
    all_image_names = [l.split('/')[-1] for l in glob.glob(os.path.join(first_folder, '{}_*.png'.format(specific_video)))]
else:
    all_image_names = [l.split('/')[-1] for l in glob.glob(os.path.join(first_folder, '*.png'))]



def generate_combined():
    for img in all_image_names:
        im1 = cv2.imread(os.path.join(first_folder, img))
        im2 = cv2.imread(os.path.join(third_folder, img))
        im_h = cv2.hconcat([im1, im2])
        cv2.imwrite(os.path.join(combined_folder, img), im_h)


def generate_video():
    all_video_and_index = {}
    for img in all_image_names:
        video_ind = img.split('_')[0]
        all_video_and_index.setdefault(video_ind, [])

    for video_ind in all_video_and_index:
        # indices = all_video_and_index[video_ind]
        # indices.sort()
        os.system("ffmpeg -r {} -i {}/{}_%d.png -vcodec mpeg4 -q:v 3 -y {}.mp4".format(RATE, combined_folder, video_ind, os.path.join(video_folder, video_ind)))
        # for i in indices:
        #     img_name = os.path.join(combined_folder, '{}_{}.png'.format(video_ind, i))

# generate_combined()
generate_video()
pdb.set_trace()
