import cv2
import os.path as osp
import glob


def images2video(file_dir='../output/exp_01-05_18:06/vis', file_pattern='InputIs*.jpg', video_name='video.avi'):
    # images to be converted to a video
    images = sorted(glob.glob(file_dir + '/' + file_pattern))
    # name for the saving video
    video_name = osp.join(file_dir, video_name)

    # assume 30 fps video
    frame = cv2.imread(osp.join(images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    # write images into the video stream
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()

    print('Video saved in ', video_name)
