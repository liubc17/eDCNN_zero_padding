import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, ZeroPadding2D, ZeroPadding1D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse


cfg = ['fc', 'cDCNN', 'cDCNN_fc', 'eDCNN', 'eDCNN_pl']

parser = argparse.ArgumentParser(description='convolution factorization experiment')
parser.add_argument('--cfg', type=str, choices=cfg, help='configuration of the network')
parser.add_argument('--repeat', type=int, default=10, help='number of experiments repeated')
args = parser.parse_args()

df_1 = pd.read_csv("../heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../heartbeat/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])


df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


def fc(depth):
    nclass = 1
    inp = Input(shape=(187,))
    x = inp
    for i in range(depth):
        x = Dense(80, activation=activations.relu)(x)

    dense = Dense(nclass, activation=activations.sigmoid)(x)

    model = models.Model(inputs=inp, outputs=dense)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def cDCNN(depth):
    nclass = 1
    inp = Input(shape=(187, 1))
    x = inp
    for i in range(depth):
        x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)

    dense = Dense(nclass, activation=activations.sigmoid)(x)

    model = models.Model(inputs=inp, outputs=dense)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def cDCNN_fc(depth):
    nclass = 1
    inp = Input(shape=(187, 1))
    x = inp
    for i in range(depth):
        x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)

    dense = Dense(64, activation=activations.relu)(x)
    dense = Dense(nclass, activation=activations.sigmoid)(dense)

    model = models.Model(inputs=inp, outputs=dense)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def eDCNN(depth):
    nclass = 1
    inp = Input(shape=(187, 1))
    x = inp
    for i in range(depth):
        x = ZeroPadding1D(padding=4)(x)
        x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)

    dense = Dense(nclass, activation=activations.sigmoid)(x)

    model = models.Model(inputs=inp, outputs=dense)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def eDCNN_pl(depth):
    nclass = 1
    inp = Input(shape=(187, 1))
    x = inp
    x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = MaxPool1D(2)(x)
    for i in range(depth-1):
        x = ZeroPadding1D(padding=4)(x)
        x = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)

    dense = Dense(nclass, activation=activations.sigmoid)(x)

    model = models.Model(inputs=inp, outputs=dense)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


Acc = []
F1 = []
for L in range(10):
    Acc_all = []
    F1_all = []
    for count in range(args.repeat):
        depth = L + 1
        if args.cfg == 'fc':
            model = fc(depth)
        elif args.cfg == 'cDCNN':
            model = cDCNN(depth)
        elif args.cfg == 'cDCNN_fc':
            model = cDCNN_fc(depth)
        elif args.cfg == 'eDCNN':
            model = eDCNN(depth)
        elif args.cfg == 'eDCNN_pl':
            model = eDCNN_pl(depth)
        else:
            raise ValueError("Please input a correct network configuration.")
        # file_path = "baseline_cnn_ptbdb.h5"
        # checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        # callbacks_list = [checkpoint, early, redonplat]  # early

        model.fit(X, Y, epochs=100, verbose=2, callbacks=[early, redonplat], validation_split=0.1)
        # model.load_weights(file_path)

        pred_test = model.predict(X_test)
        pred_test = (pred_test>0.5).astype(np.int8)

        f1 = f1_score(Y_test, pred_test)
        print("Test f1 score : %s " % f1)
        F1_all.append(f1)

        acc = accuracy_score(Y_test, pred_test)
        print("Test accuracy score : %s " % acc)
        Acc_all.append(acc)

    mean_F1 = np.mean(F1_all)
    mean_acc = np.mean(Acc_all)
    F1.append(mean_F1)
    Acc.append(mean_acc)

print('Accuracy=', Acc)
print('F1 score=', F1)
