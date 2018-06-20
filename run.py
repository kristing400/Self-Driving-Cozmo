import tensorflow as tf
import os
import cv2
import numpy as np
import sys

n_outputs = 41
padding = 2
dev = float(padding) / 3.0

keep_rate = 0.8

def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initial = initializer(shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def convolutional_neural_network(x):
    W_conv1 = weight_variable([5, 5, 3, 24])
    b_conv1 = bias_variable([24])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 2) + b_conv1)

    W_conv2 = weight_variable([5, 5, 24, 36])
    b_conv2 = bias_variable([36])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    W_conv3 = weight_variable([5, 5, 36, 48])
    b_conv3 = bias_variable([48])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

    W_fc1 = weight_variable([30*30*64, 1164])
    b_fc1 = bias_variable([1164])

    h_conv5_flat = tf.reshape(h_conv5, [-1, 30*30*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1164, 100])
    b_fc2 = bias_variable([100])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='fc2')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([100, 50])
    b_fc3 = bias_variable([50])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3, name='fc3')
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    W_fc4 = weight_variable([50, n_outputs+2*padding])
    b_fc4 = bias_variable([n_outputs+2*padding])
    output = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

    return output, h_conv1




if __name__ == "__main__":
    print(sys.argv)
    
    x = tf.placeholder('float', [None, 300,300,3])
    y = tf.placeholder('float')

    keep_prob = tf.placeholder(tf.float32)

    prediction = convolutional_neural_network(x)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, sys.argv[2])


    img_file = tf.read_file(sys.argv[1])
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.resize_images(img_decoded, [300,300])


    wei = img_decoded.eval(session=sess)

    x_input = np.array([wei])
    res = sess.run([prediction], feed_dict={x: x_input, keep_prob: 0.8})[0]

    p = res[0]

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    probs = softmax(p[0])
    zoi = 0
    for i,p in enumerate(probs):
	zoi += i * p

    zoi = max(min(zoi, n_outputs), padding) - padding

    print zoi / n_outputs

    img_res = res[1][0]

    (h,w,d) = img_res.shape

    #make grid 6x4 
    rows = 4
    cols = 6
    padding = 15
    res = np.zeros((rows*(h+padding),cols*(w+padding)))
    res.fill(255)

    print(res.shape, h, w)

    for i in range(d):
	cur_row = i / cols
	cur_col = i % cols

	print(cur_row, cur_col)

	cur_row = cur_row * (h + padding)
	cur_col = cur_col * (w + padding)

	cur_row += padding / 2
	cur_col += padding / 2

	wei = img_res[:,:,i]
	wei = wei / np.max(wei)
	wei = wei * 255

	res[cur_row:cur_row+h,cur_col:cur_col+w] = wei
	
	
    cv2.imwrite("layer1_res.png", res)



