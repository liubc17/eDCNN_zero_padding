import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

func_list = ['f1m', 'f2m']
cfg = ['fc', 'cDCNN', 'eDCNN']
bias = ['T', 'F', 'S']

parser = argparse.ArgumentParser(description='convolution factorization experiment')
parser.add_argument('--func', type=str, choices=func_list, help='simulation function')
parser.add_argument('--bias', type=str, choices=bias, default='S', help='which bias to use')
parser.add_argument('--pool', action='store_true', help='whether adding a pooling layer after the convolutional layers')
parser.add_argument('--cfg', type=str, choices=cfg, help='configuration of the network')
parser.add_argument('--edge', action='store_true', help='whether test points are drawn on the edge')
parser.add_argument('--repeat', type=int, default=10, help='number of experiments repeated')
args = parser.parse_args()

test_loss_all = []
for count in range(args.repeat):
    data_x = np.zeros((1000, 30))
    train_x = data_x[0:900, :]
    test_x = data_x[900:1000, :]

    location = np.random.randint(0, 26, 1000)

    for i in range(1000):
        data_x[i, location[i]:location[i]+5] = np.random.rand(5)

    # whether test points are drawn on the left or right side
    if args.edge:
        side_x = np.zeros((100, 30))
        for m in range(50):
            side_x[m, 0:5] = np.random.rand(5)
        for n in range(50, 100):
            side_x[n, 25:30] = np.random.rand(5)
        data_x[900:1000, :] = side_x

    data_x = np.expand_dims(data_x, 2)

    b1 = tf.Variable(0.01, dtype='double')
    b2 = tf.Variable(0.01, dtype='double')
    b3 = tf.Variable(0.01, dtype='double')
    b4 = tf.Variable(0.01, dtype='double')
    b5 = tf.Variable(0.01, dtype='double')
    b = [b1, b2, b3, b4, b5]

    if args.func == 'f1m':
        for p in range(5):
            data_x = tf.layers.conv1d(data_x, filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False,
                                      activation=tf.nn.relu, trainable=True)
            data_x = data_x + b[p]

    elif args.func == 'f2m':
        for p in range(5):
            pad1 = tf.zeros_like(data_x)[:, 0:2, :]
            x_expand = tf.concat([pad1, data_x, pad1], axis=1)
            data_x = tf.layers.conv1d(x_expand, filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False,
                                      activation=tf.nn.relu, trainable=True)
            data_x = data_x + b[p]
    else:
        raise ValueError("Please input a correct fit function.")

    out = tf.keras.layers.Flatten()(data_x)

    data_y = tf.reduce_sum(out, axis=1)
    train_y = data_y[0:900]
    test_y = data_y[900:1000]
    train_y = tf.expand_dims(train_y, 1)
    test_y = tf.expand_dims(test_y, 1)

    # tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 30])
    Y = tf.placeholder(tf.float32, [None, 1])

    if args.cfg == 'cDCNN':
        conv = tf.expand_dims(X, 2)
        if args.bias == 'T':
            for j in range(3):
                conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=True, activation=tf.nn.relu, trainable=True)
                conv = bias[j] + conv

        elif args.bias == 'F':
            for j in range(3):
                conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=tf.nn.relu, trainable=True)
                conv = bias[j] + conv

        else:
            bias1 = tf.Variable(0.01)
            bias2 = tf.Variable(0.01)
            bias3 = tf.Variable(0.01)
            bias = [bias1, bias2, bias3]
            for j in range(3):
                conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=tf.nn.relu, trainable=True)
                conv = bias[j] + conv

        conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)

        if args.pool:
            # pooling
            pool_size = 2
            pool = tf.layers.MaxPooling1D(pool_size, pool_size)
            conv = pool(conv)

        out = tf.keras.layers.Flatten()(conv)

    elif args.cfg == 'eDCNN':
        conv = tf.expand_dims(X, 2)
        if args.bias == 'T':
            for j in range(3):
                input1 = tf.zeros_like(conv)[:, 0:2, :]
                conv_expand = tf.concat([input1, conv, input1], axis=1)
                conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=True, activation=tf.nn.relu, trainable=True)

        elif args.bias == 'F':
            for j in range(3):
                input1 = tf.zeros_like(conv)[:, 0:2, :]
                conv_expand = tf.concat([input1, conv, input1], axis=1)
                conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=tf.nn.relu, trainable=True)

        else:
            bias1 = tf.Variable(0.01)
            bias2 = tf.Variable(0.01)
            bias3 = tf.Variable(0.01)
            bias = [bias1, bias2, bias3]
            for j in range(3):
                input1 = tf.zeros_like(conv)[:, 0:2, :]
                conv_expand = tf.concat([input1, conv, input1], axis=1)
                conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=tf.nn.relu, trainable=True)
                conv = bias[j] + conv

        input1 = tf.zeros_like(conv)[:, 0:2, :]
        conv_expand = tf.concat([input1, conv, input1], axis=1)
        conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)

        if args.pool:
            # pooling
            pool_size = 2
            pool = tf.layers.MaxPooling1D(pool_size, pool_size)
            conv = pool(conv)

        out = tf.keras.layers.Flatten()(conv)

    elif args.cfg == 'fc':
        fc = X
        for j in range(4):
            fc = tf.layers.dense(inputs=fc, units=10, use_bias=True, activation=tf.nn.relu, trainable=True)
        out = fc
    else:
        raise ValueError("Please input a correct network configuration.")

    pred_y = tf.layers.dense(inputs=out, units=1, use_bias=False,
                             trainable=True)

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
        training_loss, _ = sess.run([loss, optimizer], feed_dict={X: train_x, Y: train_y.eval(session=sess)})
        print("epoch {0:04d} training_loss={1:.8f}".format(
            epoch, training_loss))
    # print(sess.run(bias))
    # print(sess.run(k))

    test_loss = sess.run(loss, feed_dict={X: test_x, Y: test_y.eval(session=sess)})
    test_loss_all.append(test_loss)

mean_loss = np.mean(test_loss_all)
variance_loss = np.var(test_loss_all)
std_loss = np.sqrt(variance_loss)

print('Fit function:', args.func)
print('Network configuration:', args.cfg)
print('mean_loss:', mean_loss)
print('std_loss:', std_loss)






