import argparse

import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

cfg = ['fc', 'cDCNN', 'eDCNN']
b_list = [0., 0.01, 0.1]
noise_list = [0.01, 0.1, 1., 0.]
func_list = ['f1', 'f2', 'f3']
position = ['beginning', 'middle', 'end']

parser = argparse.ArgumentParser(description='simulation experiment')
parser.add_argument('--cfg', type=str, choices=cfg, help='configuration of the network')
parser.add_argument('--func', type=str, choices=func_list, help='simulation function')
parser.add_argument('--pos', type=str, default='middle', choices=position, help='data position of f3')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--pl', action='store_true', help='whether adding pooling after conv module')
parser.add_argument('--pl_size', type=int, default=2, help='pool size')
parser.add_argument('--fc', action='store_true', help='whether adding fully connected layer after conv module')
parser.add_argument('--init_b', type=float, default=0.01, choices=b_list, help='initial bias for cDCNN or eDCNN')
parser.add_argument('--noise', type=float, default=0.1, choices=noise_list, help='noise for learning')
parser.add_argument('--num_train', type=int, default=1000, help='number of training samples')
parser.add_argument('--num_test', type=int, default=100, help='number of test samples')
parser.add_argument('--repeat', type=int, default=10, help='number of experiments repeated')

args = parser.parse_args()

test_loss_all = []
for count in range(args.repeat):
    if args.func == 'f1':
        dim = 5
        kernels = 1
        # training data
        trainX = -1 + 2 * np.random.rand(args.num_train, 5)
        trainY = np.expand_dims(trainX[:, 0] - trainX[:, 1] + trainX[:, 2] * trainX[:, 3] + trainX[:, 4] ** 2, 1)

        # test data
        testX = -1 + 2 * np.random.rand(args.num_test, 5)
        testY = np.expand_dims(testX[:, 0] - testX[:, 1] + testX[:, 2] * testX[:, 3] + testX[:, 4] ** 2, 1)

    elif args.func == 'f2':
        dim = 30
        kernels = 1
        # training data
        trainX = np.zeros((args.num_train, 30))
        location = np.random.randint(0, 26, args.num_train)
        for i in range(args.num_train):
            trainX[i, location[i]:location[i] + 5] = -1 + 2 * np.random.rand(5)
        trainY = np.zeros(args.num_train)
        for i in range(26):
            trainY += trainX[:, i] * trainX[:, i+1] * trainX[:, i+2] * trainX[:, i+3] * trainX[:, i+4]
        trainY = np.expand_dims(trainY, 1)

        # test data
        testX = np.zeros((args.num_test, 30))
        location2 = np.random.randint(0, 26, args.num_test)
        for i in range(args.num_test):
            testX[i, location2[i]:location2[i] + 5] = -1 + 2 * np.random.rand(5)
        testY = np.zeros(args.num_test)
        for i in range(26):
            testY += testX[:, i] * testX[:, i+1] * testX[:, i+2] * testX[:, i+3] * testX[:, i+4]
        testY = np.expand_dims(testY, 1)

    elif args.func == 'f3':
        dim = 30
        kernels = 10
        # training data
        trainX = np.zeros((args.num_train, dim))
        if args.pos == 'beginning':
            key = (0, 1, 2, 3, 4)
        elif args.pos == 'end':
            key = (25, 26, 27, 28, 29)
        else:
            key = (13, 14, 15, 16, 17)
        for i in range(args.num_train):
            trainX[i, key] = np.random.rand(5)
        trainX_norm = np.linalg.norm(trainX, ord=2, axis=1) ** 2
        trainY = np.expand_dims(np.sin(trainX_norm) + 0.5 * np.cos(trainX_norm) +
                                np.random.randn(args.num_train) * args.noise, 1)

        # test data
        testX = np.zeros((args.num_test, dim))
        for u in range(args.num_test):
            testX[u, key] = np.random.rand(5)
        testX_norm = np.linalg.norm(testX, ord=2, axis=1) ** 2
        testY = np.expand_dims(np.sin(testX_norm) + 0.5 * np.cos(testX_norm), 1)
    else:
        raise ValueError("Please input a correct fit function.")

    # tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    if args.cfg == 'fc':
        fc = X
        for j in range(args.depth):
            fc = tf.layers.dense(inputs=fc, units=10, use_bias=True, activation=tf.nn.relu, trainable=True)

        pred_y = tf.layers.dense(inputs=fc, units=1, use_bias=False,
                                 trainable=True)

    elif args.cfg == 'cDCNN':
        conv = tf.expand_dims(X, 2)
        bias = []
        for q in range(args.depth-1):
            bias.append(tf.Variable(args.init_b))
        for j in range(args.depth-1):
            conv = tf.layers.conv1d(conv, filters=kernels, kernel_size=3, strides=1, padding='valid',
                                    use_bias=False, activation=tf.nn.relu, trainable=True)
            conv = bias[j] + conv

        conv = tf.layers.conv1d(conv, filters=kernels, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)

        out = tf.keras.layers.Flatten()(conv)
        if args.fc:
            out = tf.layers.dense(inputs=out, units=out.shape[1], use_bias=True, activation=tf.nn.relu, trainable=True)

        pred_y = tf.layers.dense(inputs=out, units=1, use_bias=False,
                                 trainable=True)

    elif args.cfg == 'eDCNN':
        conv = tf.expand_dims(X, 2)
        bias = []
        for q in range(args.depth-1):
            bias.append(tf.Variable(args.init_b))
        for j in range(args.depth-1):
            input1 = tf.zeros_like(conv)[:, 0:2, :]
            conv_expand = tf.concat([input1, conv, input1], axis=1)
            conv = tf.layers.conv1d(conv_expand, filters=kernels, kernel_size=3, strides=1, padding='valid',
                                    use_bias=False, activation=tf.nn.relu, trainable=True)
            conv = bias[j] + conv

        input1 = tf.zeros_like(conv)[:, 0:2, :]
        conv_expand = tf.concat([input1, conv, input1], axis=1)
        conv = tf.layers.conv1d(conv_expand, filters=kernels, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)

        # pooling
        if args.pl:
            pool_size = args.pl_size
            pool = tf.layers.MaxPooling1D(pool_size, pool_size)
            conv = pool(conv)

        out = tf.keras.layers.Flatten()(conv)
        pred_y = tf.layers.dense(inputs=out, units=1, use_bias=False,
                                 trainable=True)
    else:
        raise ValueError("Please input a correct network configuration.")

    loss = tf.sqrt(tf.reduce_mean(tf.multiply(Y - pred_y, Y - pred_y)))

    global_step = tf.Variable(0, trainable=False)

    boundaries = [800, 1200, 1500]
    values = [0.003, 0.001, 0.0003, 0.0001]
    epochs = 2000
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        training_loss, _ = sess.run([loss, optimizer], feed_dict={X: trainX, Y: trainY})
        print("epoch {0:04d} training_loss={1:.8f}".format(
            epoch, training_loss))

    test_loss = sess.run(loss, feed_dict={X: testX, Y: testY})
    test_loss_all.append(test_loss)

mean_loss = np.mean(test_loss_all)
variance_loss = np.var(test_loss_all)
std_loss = np.sqrt(variance_loss)

print('function is', args.func)
if args.func == 'f3':
    print('noise is', args.noise)
if not args.cfg == 'fc':
    print('initial bias is', args.init_b)
if args.pl:
    args.cfg = args.cfg + '+pl'
    print('pool size is', args.pl_size)
if args.fc:
    args.cfg = args.cfg + '+fc'
print('network is', args.cfg)
print('network depth is', args.depth)
print('mean_loss is', mean_loss)
print('std_loss:', std_loss)






