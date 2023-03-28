# -*- coding: utf-8 -*-


# Compatibility layer between Python 2 and Python 3
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import tensorflow as tf

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

import os
import argparse

# %%

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cfg = ['fc', 'cDCNN', 'eDCNN']

parser = argparse.ArgumentParser(description='simulation experiment')
parser.add_argument('--cfg', type=str, choices=cfg, help='configuration of the network')
parser.add_argument('--pl_size', type=int, default=0, help='pool size')
parser.add_argument('--fc', type=int, default=0, help='whether adding fully connected layer after conv module')
parser.add_argument('--repeat', type=int, default=10, help='number of experiments repeated')
args = parser.parse_args()


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe,
                              preview_rows=20):
    """
    This function shows basic information for the given dataframe

    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview

    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())


def read_data(file_path):
    """
    This function reads the accelerometer data from a file

    Args:
        file_path: URL pointing to the CSV file

    Returns:
        A pandas dataframe
    """

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


# Not used right now
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
                                        figsize=(15, 10),
                                        sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def create_segments_and_labels(df, time_steps, step, label_name):
    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels

    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

LABELS = ["Downstairs",
          "Jogging",
          "Sitting",
          "Standing",
          "Upstairs",
          "Walking"]
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('Data/WISDM_ar_v1.1_raw.txt')

# Describe the data
show_basic_dataframe_info(df, 20)

df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()

df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()

for activity in np.unique(df["activity"]):
    subset = df[df["activity"] == activity][:180]
    plot_activity(activity, subset)

# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df["activity"].values.ravel())

# %%

print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
df_test = df[df['user-id'] > 28]
df_train = df[df['user-id'] <= 28]

# Normalize features for training data set
df_train['x-axis'] = feature_normalize(df['x-axis'])
df_train['y-axis'] = feature_normalize(df['y-axis'])
df_train['z-axis'] = feature_normalize(df['z-axis'])
# Round in order to comply to NSNumber from iOS
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

# %%

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (20869, 40, 3)
print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods * num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
# x_train shape: (20869, 120)
print('input_shape:', input_shape)
# input_shape: (120)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

print("\n--- Check against test data ---\n")

# Normalize features for test data set
df_test['x-axis'] = feature_normalize(df_test['x-axis'])
df_test['y-axis'] = feature_normalize(df_test['y-axis'])
df_test['z-axis'] = feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

print("\n--- Create neural network model ---\n")


mean_all = []
std_all = []
mean_all_loss = []
std_all_loss = []
for j in range(9):
    L = j + 1

    testaccuracy_our = []
    loss_our = []
    for k in range(args.repeat):
        print('L= ', L, ', i= ', k, '!')

        if args.cfg == 'fc':
            keras.backend.clear_session()

            input = tf.keras.layers.Input(shape=[240])
            x = tf.keras.layers.Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,))(input)
            for i in range(L):
                x = tf.keras.layers.Dense(60)(x)
                x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(6)(x)
            x = tf.keras.layers.Activation('softmax')(x)
            model_m = tf.keras.models.Model(input, x)
            #         print(model_m.summary())

        elif args.cfg == 'cDCNN':
            kernelsize = 9
            filternumber = 20
            keras.backend.clear_session()

            # tf.reset_default_graph()

            input = tf.keras.layers.Input(shape=[240])
            x = tf.keras.layers.Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,))(input)
            for i in range(L - 1):
                x = tf.keras.layers.Conv1D(filternumber, kernelsize, use_bias=False)(x)
                x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Conv1D(filternumber, kernelsize)(x)
            x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Flatten()(x)
            if args.fc:
                x = tf.keras.layers.Dense(args.fc)(x)

            x = tf.keras.layers.Dense(6)(x)
            x = tf.keras.layers.Activation('softmax')(x)
            model_m = tf.keras.models.Model(input, x)
            #         print(model_m.summary())

        elif args.cfg == 'eDCNN':
            kernelsize = 9
            filternumber = 20
            keras.backend.clear_session()

            # tf.reset_default_graph()
            def expand_input(x, kernelsize):
                input1 = tf.zeros_like(x)[:, 0:kernelsize - 1, :]
                y = tf.concat([input1, x, input1], axis=1)
                return y

            input = tf.keras.layers.Input(shape=[240])
            x = tf.keras.layers.Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,))(input)
            for i in range(L - 1):
                x = tf.keras.layers.Lambda(expand_input, arguments={'kernelsize': kernelsize})(x)
                x = tf.keras.layers.Conv1D(filternumber, kernelsize, use_bias=False)(x)
                x = tf.keras.layers.Activation('relu')(x)

            x = tf.keras.layers.Lambda(expand_input, arguments={'kernelsize': kernelsize})(x)
            x = tf.keras.layers.Conv1D(filternumber, kernelsize)(x)
            x = tf.keras.layers.Activation('relu')(x)

            if args.pl_size:
                x = MaxPooling1D(args.pl_size)(x)
            x = tf.keras.layers.Flatten()(x)

            x = tf.keras.layers.Dense(6)(x)
            x = tf.keras.layers.Activation('softmax')(x)
            model_m = tf.keras.models.Model(input, x)
            #         print(model_m.summary())

        else:
            raise ValueError("Please input a correct network configuration.")

        # %%

        print("\n--- Fit the model ---\n")

        # The EarlyStopping callback monitors training accuracy:
        # if it fails to improve for two consecutive epochs,
        # training stops early
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

        model_m.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        # Hyper-parameters
        BATCH_SIZE = 512
        EPOCHS = 150

        # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
        history = model_m.fit(x_train,
                              y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              #                       callbacks=callbacks_list,
                              callbacks=None,
                              validation_split=0.2,
                              verbose=1)

        print("\n--- Check against test data ---\n")

        # Set input_shape / reshape for Keras
        x_test = x_test.reshape(x_test.shape[0], input_shape)

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32")

        score = model_m.evaluate(x_test, y_test, verbose=1)

        print("\nAccuracy on test data: %0.4f" % score[1])
        print("\nLoss on test data: %0.4f" % score[0])
        testaccuracy_our.append(score[1])
        loss_our.append(score[0])
    mean_our = np.mean(testaccuracy_our)
    mean_our = round(mean_our, 4)
    variance_our = np.var(testaccuracy_our)
    std_our = np.sqrt(variance_our)
    mean_all.append(mean_our)
    std_all.append(std_our)

    mean_loss = np.mean(loss_our)
    mean_loss = round(mean_loss, 4)
    variance_loss = np.var(loss_our)
    std_loss = np.sqrt(variance_loss)
    mean_all_loss.append(mean_loss)
    std_all_loss.append(std_loss)

if args.pl_size:
    args.cfg = args.cfg + '+pl' + str(args.pl_size)
    print('pool size is', args.pl_size)
if args.fc:
    args.cfg = args.cfg + '+fc'

plt.errorbar(np.arange(9) + 1, mean_all, yerr=std_all, fmt='x:', ecolor='r', color='b', elinewidth=2, capsize=4)
# plt.plot(np.arange(9)+2,mean_ori)
plt.xlabel('depth L,'+args.cfg)
plt.ylabel('acc')
plt.show()

plt.errorbar(np.arange(9) + 1, mean_all_loss, yerr=std_all_loss, fmt='x:', ecolor='r', color='b', elinewidth=2,
             capsize=4)
# plt.plot(np.arange(9)+2,mean_ori)
plt.xlabel('depth L,'+args.cfg)
plt.ylabel('loss')
plt.show()

print('acc:', mean_all)
print('loss:', mean_all_loss)
