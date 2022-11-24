import os
import cv2
import numpy as np

import chessPoly
import checkPos

PROD_EPS: float = 30.0
lineCount: int = 0


def get_chess_line(input_image):
    """
    It takes an image, converts it to grayscale, finds the edges, and then finds lines in the image

    :param input_image: the image we're searching in
    :return: a list of lines. Each line is represented by a list of 4 numbers.
    """
    image = input_image.copy()
    # img_size = image.shape

    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_length = PROD_EPS
    max_line_gap = 3

    # creating an image copy to draw lines on
    # line_image = np.copy(image)

    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    return chessPoly.remove_same_line(lines)


def split_frames_mp4(source_file_name):
    """
    The function `split_frames_mp4` takes a video file and splits it into frames, and then displays the frames in real
    time

    :param source_file_name: the name of the video file, which is the name of the video file without the suffix
    """

    video_path = os.path.join('./asserts/', source_file_name + '.mp4')
    times = 0

    # get the image from the video, with the frequency of frame_frequency
    frame_frequency = 1

    camera = cv2.VideoCapture(video_path)
    # frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize the variable
    lines = []
    line_count = 0
    feature = 255 * 3 // 2

    while True:
        times += 1
        res, image = camera.read()
        if not res:
            # print('not res , not image')
            break

        if line_count != 38:
            line_count, lines = get_chess_line(image)
            feature = checkPos.get_board_avg_color(image, lines)

        if times % frame_frequency == 0:
            # cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
            go_map, image = checkPos.trans_go_map(image, lines, feature)
            # time_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # cv2.putText(image, time_text, (word_x,word_y),
            # cv2.FONT_HERSHEY_SIMPLEX,1,(55,255,155),2)
            cv2.imshow("real_time", image)
            # print(outPutDirName + str(times)+'.jpg')

        # cv2.imwrite(outPutDirName + str(times) + '.jpg', image)
        # print(times)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()


if __name__ == '__main__':
    im_file = "./asserts/"

    for im_name in os.listdir(im_file):
        suffix_file = os.path.splitext(im_name)[-1]
        if suffix_file == '.mp4':
            sourceFileName = os.path.splitext(im_name)[0]
            split_frames_mp4(sourceFileName)
