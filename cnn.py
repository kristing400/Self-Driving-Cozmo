import tensorflow as tf
import os
import cv2
import numpy as np
import math
from scipy.stats import norm

n_outputs = 41
padding = 2
dev = float(padding) / 3.0

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    images = os.listdir(directory)
    filenames = []
    labels = []

    x_temp = np.arange(0, n_outputs + 2 * padding)

    for f in images:
        label = float(f.split("_")[1][:-4])
	label = label * n_outputs + 2 * padding
	dist = norm.pdf(x_temp, loc=label, scale=dev)
	filenames.append(os.path.join(directory, f))
	labels.append(dist)

    return np.array(filenames), np.array(labels)

batch_size = 64

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
    print(h_conv5)

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



    return output



def _input_parser(img_path, label):
    #return img_path, label
    # read the img from file
    img_file = tf.read_file(img_path)
    #img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.decode_png(img_file, channels=3)
    #print(img_decoded.shape)

    img_decoded = tf.image.resize_images(img_decoded, [300,300])
    #img_decoded = tf.image.resize_images(img_decoded, [66,200])
    #img_decoded = tf.reshape(img_decoded, shape=[300 * 300 * 3])

    #label = tf.cast(tf.floor(label / (1.0/n_outputs)), dtype=tf.int32)
    #label = tf.one_hot(label, n_outputs)

    #label = label * n_outputs + padding
    #label = label.eval(session=sess)
    #x_temp = np.arange(0, n_outputs + 2 * padding)

    #print(type(x_temp), type(label))

    return img_decoded, label #dist #label #tf.convert_to_tensor([label])

#train_filenames, train_labels = list_images("weija/train")
#val_filenames, val_labels = list_images("weija/val")

#train_filenames, train_labels = list_images("cozmo_training/train_set")
train_filenames, train_labels = list_images("may-2-data")
#train_filenames, train_labels = list_images("data_1")
val_filenames, val_labels = list_images("new_val_data")


train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
train_dataset = train_dataset.map(_input_parser)
batched_train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels)))
val_dataset = val_dataset.map(_input_parser)
batched_val_dataset = val_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
train_init_op = iterator.make_initializer(batched_train_dataset)
val_init_op = iterator.make_initializer(batched_val_dataset)


# iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
# train_init_op = iterator.make_initializer(train_dataset)
# val_init_op = iterator.make_initializer(val_dataset)

# Indicates whether we are in training or in test mode
is_training = tf.placeholder(tf.bool)

next_element = iterator.get_next()

# x = tf.placeholder('float', [None, 270000])
# x = tf.placeholder('float', [None, 784])

#x = tf.placeholder('float', [None, 120, 160, 3])
x = tf.placeholder('float', [None, 300, 300, 3])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    #cost = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    #cost = tf.reduce_mean(tf.square(y - prediction))
    #accuracy = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=prediction))
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("tensorboard-logs", graph=tf.get_default_graph())


    with tf.name_scope("loss"):
        my_loss = tf.placeholder(tf.float32, name="loss")
    tf.summary.scalar("loss", my_loss)
    summary_op = tf.summary.merge_all()

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def test():

        sess.run(val_init_op)
        total = 0
        score = 0
        first = False
        while True:
            try:
                epoch_x, epoch_y = sess.run(next_element)
                (n, _) = epoch_y.shape
                p = sess.run([prediction], feed_dict={x: epoch_x, keep_prob: 0.8})

                if not first:
                    probs = softmax(p[0][0])
		    zoi = 0
		    for i,p in enumerate(probs):
			zoi += i * p

		    zoi = max(min(zoi, n_outputs), padding) - padding
			
		    print(probs, zoi, np.argmax(epoch_y, axis=1)[0])
		    first = True
	    except tf.errors.OutOfRangeError:
		break

	return 0


    optimizer = tf.train.AdamOptimizer().minimize(cost)


    sess.run(tf.global_variables_initializer())

    hm_epochs = 50

    for epoch in range(hm_epochs):
        epoch_loss = 0
	total = 0

        sess.run(train_init_op)
        while True:
            try:
                epoch_x, epoch_y = sess.run(next_element)
		(n, _) = epoch_y.shape
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.8})
                epoch_loss += c
                total += n

            except tf.errors.OutOfRangeError:
                break

	test()

	save_path = saver.save(sess, "ckpts/model_%d.ckpt" % (epoch+1))

        summary = sess.run(summary_op, feed_dict={my_loss: (epoch_loss/total)})
	writer.add_summary(summary, epoch+1)
        print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss/total, save_path)

    test()
    save_path = saver.save(sess, "ckpts/final_model.ckpt")

train_neural_network(x)
