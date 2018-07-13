#!/usr/bin/env python3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#           @title:         Attacking AlexNet (a convolutional neural network for object identification) 
#           @author:      Ron Urbach
#           @date:         June 2018
#           @notes:       
#                   - Assumes "bvlc_alexnet.npy", "caffe_classes", "truck.png" files are in the
#                        relative running directory.
#                   - AlexNet code based on: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#                   - This code was used in the (hebrew) article: 
#                        https://www.digitalwhisper.co.il/files/Zines/0x60/DW96-1-ToasterTruck.pdf
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import tensorflow as tf
from caffe_classes import class_names
    
VALID_PAD = 'VALID'
SAME_PAD = 'SAME'


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding=VALID_PAD, group=1):
    """ generates a convolutional layer for the neural network """
    c_i = input.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3) 
        kernel_groups = tf.split(kernel, group, 3) 
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def alexnet(net_data, x):
    """ defines the architecture of Alexnet """
    with tf.name_scope('conv1'):
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding=SAME_PAD, group=1)
        conv1 = tf.nn.relu(conv1_in)
    with tf.name_scope('lrn1'):
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    with tf.name_scope('maxpool1'):
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=VALID_PAD)
    with tf.name_scope('conv2'):
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding=SAME_PAD, group=2)
        conv2 = tf.nn.relu(conv2_in)
    with tf.name_scope('lrn2'):
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    with tf.name_scope('maxpool2'):
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=VALID_PAD)
    with tf.name_scope('conv3'):
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding=SAME_PAD, group=1)
        conv3 = tf.nn.relu(conv3_in)
    with tf.name_scope('conv4'):
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding=SAME_PAD, group=2)
        conv4 = tf.nn.relu(conv4_in)
    with tf.name_scope('conv5'):
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding=SAME_PAD, group=2)
        conv5 = tf.nn.relu(conv5_in)
    with tf.name_scope('maxpool5'):
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=VALID_PAD)
    with tf.name_scope('fc6'):
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(maxpool5_flat, fc6W, fc6b)
    with tf.name_scope('fc7'):
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    with tf.name_scope('fc8'):
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    prob = tf.nn.softmax(fc8)
    return prob, fc8

    
def inference(img_path, net_data, top_amount=5):
    """ runs the alexnet on the given image from path (img_path) with the pretrained weights (net_data) """
    img = (imread(img_path)[:, :, :3]).astype(np.float64)
    img -= np.mean(img)
    img = img[:, :, ::-1]
    input_shape = (227, 227, 3)
    x = tf.placeholder(tf.float32, (None,) + input_shape)
    prob, fc8 = alexnet(net_data, x)
    init = tf.global_variables_initializer() 
    sess = tf.Session()
    sess.run(init)
    output = sess.run(prob, feed_dict={x: [img]})  # could run on many images
    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind, :]
        print("Image", input_im_ind)
        for i in range(top_amount):
            print(inds[-1 - i], class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]])


def attack(original_img, net_data, fake_class, original_class, train_iterations=50):
    """ fools the neural network by optimizing input toward fake_class classification """
    input_shape = list(original_img.shape)
    init_var_input_image = tf.constant(original_img.astype(np.float32).reshape([1] + input_shape))
    input_image_var_unchanged = tf.get_variable("input_image_variable", initializer=init_var_input_image)
    black_img = np.zeros(input_shape, dtype=np.float64)
    x = tf.placeholder(tf.float32, [None]+ input_shape)
    initial_image = tf.get_variable("initial_image_variable", initializer=init_var_input_image)
    opt_im_var = tf.Variable(initial_image)
    opt_x = x + opt_im_var
    prob, fc8 = alexnet(net_data, opt_x)
    loss = -fc8[0, fake_class]  # optimize into fake class
    penalty_on_input_change = (tf.reduce_sum(tf.square(tf.subtract(opt_im_var, input_image_var_unchanged))))
    loss += 0.00003 * penalty_on_input_change  # regularize
    
    train_step = tf.train.AdamOptimizer(0.95).minimize(loss, var_list=[opt_im_var])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(train_iterations):
        var_mean = np.mean(opt_im_var.eval(session=sess))
        output = sess.run(prob, feed_dict={x: [black_img - var_mean]})
        sess.run(train_step, feed_dict={x: [black_img]})
    print("fake prob:",output[0, fake_class], "\toriginal prob:",output[0, original_class])
    fake = opt_im_var.eval(session=sess)[0, :, :, ::-1]
    for channel in range(3):
        fake[:,:,channel] -= fake[:,:,channel].min()
        fake[:,:,channel] /= fake[:,:,channel].max()
    plt.imshow(fake)
    plt.show()

    
def main():
    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    img = (imread('truck.png')[:, :, :3]).astype(np.float64)
    img -= np.mean(img)
    img = img[:, :, ::-1]
    attack(img, net_data, 859, 867)  # 859=toaster class id, 867=truck class id
    
    
if __name__ == "__main__":
    main()
 