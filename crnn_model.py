"""Keras implementation of CRNN."""

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPool2D, BatchNormalization, LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Reshape, Permute, Lambda
from keras.layers.advanced_activations import LeakyReLU


def CRNN(input_shape, num_classes, prediction_only=False, gru=False, cnn=False):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
    """
    
    #act = LeakyReLU(alpha=0.3)
    act = 'relu'
    
    x = image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='batchnorm1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    x = BatchNormalization(name='batchnorm2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool5', padding='valid')(x)
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)
    s = x.shape.as_list()
    x = Reshape((s[1],s[3]))(x)
    
    if cnn:
        for i in range(6):
            x = BatchNormalization()(x)
            x1 = Conv1D(128, 5, strides=1, dilation_rate=1, padding='same', activation=act)(x)
            x2 = Conv1D(128, 5, strides=1, dilation_rate=2, padding='same', activation=act)(x)
            x = concatenate([x1,x2])
    elif gru:
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Bidirectional(GRU(256, return_sequences=True))(x)
    else:
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    x = Dense(num_classes, name='dense1')(x)
    x = y_pred = Activation('softmax', name='softmax')(x)
    
    model_pred = Model(image_input, x)
    
    if prediction_only:
        return model_pred

    max_string_len = s[1]
    
    # since keras currently does not support loss functions with extra parameters, 
    # we put the CTC loss in a lambda layer and call compile with a dummy loss
    def ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    labels = Input(name='label_input', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

    model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=ctc_loss)
    
    return model_train, model_pred
    
