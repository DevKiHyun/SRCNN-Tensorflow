import argparse
import tensorflow as tf
import numpy as np
import cv2
import os

import SRCNN.srcnn as srcnn
from SRCNN.util import ImageBatch
from SRCNN.util import display
from SRCNN.util import psnr

def modcrop(image, scale):
    height, width, _ = image.shape
    height = height - (height % scale)
    width = width - (width % scale)
    image = image[0:height, 0:width, :]

    return image

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the data type

def bicubic_sr(low_input, scale):
    upscaled_rgb = np.clip(cv2.resize(low_input.copy(), None, fx=1.0 * scale, fy=1.0 * scale, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

    return upscaled_rgb

def SRCNN_sr(sess, SRCNN, low_input, scale):
    upscaled_rgb = np.clip(cv2.resize(low_input.copy(), None, fx=1.0 * scale, fy=1.0 * scale, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    upscaled_ycrcb = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2YCR_CB)

    y_ch = upscaled_ycrcb[:,:,0].copy()
    y_ch = im2double(y_ch.astype(np.uint8))

    srcnn_y_ch = sess.run(SRCNN.output, feed_dict={SRCNN.X:np.expand_dims([y_ch], axis=-1)})[0]

    upscaled_ycrcb[:, :, 0] = srcnn_y_ch[:,:,0]*255
    rgb = np.clip(cv2.cvtColor(upscaled_ycrcb, cv2.COLOR_YCR_CB2RGB), 0, 255)

    return rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channel', type=int, default=1, help='-')
    parser.add_argument('--batch_size', type=int, default=100, help='-')
    parser.add_argument('--image_index', type=int, default=7, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    parser.add_argument('--coordinate', type=int, default=[50, 50], help='-')
    parser.add_argument('--interval', type=int, default=30, help='-')
    args, unknown = parser.parse_known_args()

    test_y_images_path = './data/Urban100/HR/*.png'
    result_save_path = './test_result'
    if not os.path.exists(result_save_path): os.makedirs(result_save_path)

    labels_images = ImageBatch(test_y_images_path, training_ratio=1, on_sort=True, ext='png')
    labels = labels_images.next_batch(batch_size=args.batch_size)

    SRCNN = srcnn.SRCNN(args)
    SRCNN.neuralnet()

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, './model/SRCNN.ckpt')

    index = args.image_index
    scale = args.scale
    x_start = args.coordinate[0]
    y_start = args.coordinate[1]
    interval = args.interval

    label = modcrop(labels[index].copy(), scale=scale)

    low_rs = np.clip(cv2.resize(label.copy(), None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC), 0, 255)

    bicubic_output = bicubic_sr(low_rs.copy(), scale=scale)
    SRCNN_output = SRCNN_sr(sess, SRCNN, low_rs.copy(), scale=scale)

    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'original', scale), label)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'low', scale), low_rs)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'bicubic', scale), bicubic_output)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'SRCNN', scale), SRCNN_output)

    print("Bicubic {}x PSNR: ".format(scale), psnr(label, bicubic_output))
    print("SRCNN {}x PSNR: ".format(scale), psnr(label, SRCNN_output))

    #input_list = [input, input[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #bicubic_list = [bicubic_output, bicubic_output[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #SRCNN_list = [SRCNN_output, SRCNN_output[x_start:x_start+interval,  y_start:y_start+interval, :]]

    original_list = np.array([label, bicubic_output, SRCNN_output])

    #zoom_list = np.array(original_list[:])
    display_list = np.array([original_list])
    display(display_list)

    #display_list = np.array([original_list, zoom_list)
    #display(display_list,  figsize = (5,5), axis_off=True, size_equal=True, gridspec=(0,0), zoom_coordinate=(150, 190, 100,260))