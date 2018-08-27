# This code trains a CNN for character recognitiom
import MyImageLoader
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, len(MyImageLoader.Y[0])])

input_ = 3 * 3 * 1
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W = tf.get_variable("W", (3, 3, 1, 64), tf.float32, initializer)
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))

conv1 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b))
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

input_ = 3 * 3 * 64
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

conv2_out = tf.nn.leaky_relu(tf.nn.bias_add(conv2, b2))
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2_normalize = tf.reshape(max_pool2, shape=[-1, 3 * 3 * 32])
input_ = 3 * 3 * 32
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W3 = tf.get_variable("W3", (3 * 3 * 32, 500), tf.float32, initializer)
b3 = tf.get_variable("b3", [500], tf.float32, tf.constant_initializer(0))

f1_output = tf.nn.tanh(tf.matmul(conv2_normalize, W3) + b3)

input_ = 500
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W4 = tf.get_variable("W4", (500, len(MyImageLoader.Y[0])), tf.float32, initializer)
b4 = tf.get_variable("b4", [len(MyImageLoader.Y[0])], tf.float32, tf.constant_initializer(0))

f2_output = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(f1_output, W4) + b4))
global_step = tf.Variable(0, name="global_step", trainable=False)


entropy = y * tf.log(f2_output)
cross_entropy = -tf.reduce_sum(entropy)
loss = tf.reduce_sum(cross_entropy)

tf.summary.scalar("cost", loss)
summary_op = tf.summary.merge_all()


optimizer = tf.train.AdamOptimizer()

minimize_loss = optimizer.minimize(loss)


train_saver = tf.train.Saver()

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.initialize_all_variables())
summary_writer = tf.summary.FileWriter("summaries", session.graph)

for iteration in range(0, 250):
    #res = session.run(conv2_out, feed_dict={x: MyImageLoader.X, y: MyImageLoader.Y})
    #print "conv2 Res shape : ", res.shape
   # res = session.run(max_pool2, feed_dict={x: MyImageLoader.X, y: MyImageLoader.Y})
    #print "max pool 2 Res shape : ", res.shape
    for index in range(0, len(MyImageLoader.X), 20):

        session.run(minimize_loss, feed_dict={x: MyImageLoader.X[index:index+20], y: MyImageLoader.Y[index:index+20]})
    print("Iteration %d - Loss : %f" % (iteration, session.run(loss, feed_dict={x: MyImageLoader.X, y: MyImageLoader.Y})))
import os

train_saver.save(session, "my-CNN-test-model/my-CNN-test-model")