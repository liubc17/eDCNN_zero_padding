import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

func_list = ['f1', 'f2', 'f3']
cfg = ['fc', 'cDCNN', 'eDCNN']

parser = argparse.ArgumentParser(description='convolution factorization experiment')
parser.add_argument('--func', type=str, choices=func_list, help='simulation function')
parser.add_argument('--cfg', type=str, choices=cfg, help='configuration of the network')
parser.add_argument('--block', type=int, default=1, help='number of blocks of the chosen cfg')
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
    data_x = np.expand_dims(data_x, 2)

    if args.func == 'f1':
        for p in range(5):
            data_x = tf.layers.conv1d(data_x, filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False,
                                      activation=tf.nn.relu, trainable=True)
    elif args.func == 'f2':
        for p in range(5):
            pad1 = tf.zeros_like(data_x)[:, 0:2, :]
            x_expand = tf.concat([pad1, data_x, pad1], axis=1)
            data_x = tf.layers.conv1d(x_expand, filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False,
                                      activation=tf.nn.relu, trainable=True)
    elif args.func == 'f3':
        for p in range(5):
            data_x = tf.layers.dense(inputs=data_x, units=5, use_bias=True, activation=tf.nn.relu, trainable=True)
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

    if args.cfg == 'eDCNN':
        # multi-conv eDCNN
        conv = tf.expand_dims(X, 2)
        for j in range(4):
            input1 = tf.zeros_like(conv)[:, 0:2, :]
            conv_expand = tf.concat([input1, conv, input1], axis=1)
            conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                    use_bias=False, activation=None, trainable=True)

        input1 = tf.zeros_like(conv)[:, 0:2, :]
        conv_expand = tf.concat([input1, conv, input1], axis=1)
        conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)

        # pooling
        pool_size = 4
        pool = tf.layers.MaxPooling1D(pool_size, pool_size)
        conv = pool(conv)

        if args.block == 2:
            for q in range(4):
                input1 = tf.zeros_like(conv)[:, 0:2, :]
                conv_expand = tf.concat([input1, conv, input1], axis=1)
                conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=None, trainable=True)

            input1 = tf.zeros_like(conv)[:, 0:2, :]
            conv_expand = tf.concat([input1, conv, input1], axis=1)
            conv = tf.layers.conv1d(conv_expand, filters=1, kernel_size=3, strides=1, padding='valid',
                                    use_bias=True, activation=tf.nn.relu, trainable=True)

            # pooling
            pool_size2 = 2
            pool2 = tf.layers.MaxPooling1D(pool_size2, pool_size2)
            conv = pool2(conv)

    elif args.cfg == 'cDCNN':
        # multi-conv cDCNN
        conv = tf.expand_dims(X, 2)
        for j in range(4):
            conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                    use_bias=False, activation=None, trainable=True)

        conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                use_bias=True, activation=tf.nn.relu, trainable=True)
        if args.block == 2:
            for q in range(4):
                conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                        use_bias=False, activation=None, trainable=True)

            conv = tf.layers.conv1d(conv, filters=1, kernel_size=3, strides=1, padding='valid',
                                    use_bias=True, activation=tf.nn.relu, trainable=True)

    elif args.cfg == 'fc':
        # fc
        conv = X
        conv = tf.layers.dense(inputs=conv, units=10, use_bias=True, activation=tf.nn.relu, trainable=True)
        if args.block == 2:
            conv = tf.layers.dense(inputs=conv, units=10, use_bias=True, activation=tf.nn.relu, trainable=True)
    else:
        raise ValueError("Please input a correct network configuration.")

    out = tf.keras.layers.Flatten()(conv)
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
        print("epoch {0:04d} training_loss={1:.8f}".format(epoch, training_loss))
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




