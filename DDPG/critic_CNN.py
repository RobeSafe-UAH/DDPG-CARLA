import keras.backend as keras_backend
import tensorflow as tf
from keras.layers import Dense, Input, merge, add, Conv2D, Concatenate, Flatten, AveragePooling2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from carla_config import IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS, hidden_units, FILTERS_CONV, KERNEL_CONV, POOL_SIZE, POOL_STRIDES, image_network

class CriticNetwork_CNN:
    def __init__(self, tf_session, tau=0.001, lr=0.001):
        self.tf_session = tf_session
        self.tau = tau
        self.lr = lr
        keras_backend.set_session(tf_session)

        self.model, self.state_input, self.action_input = self.generate_model()
        self.target_model, _, _ = self.generate_model()

        self.critic_gradients = tf.gradients(self.model.output, self.action_input)
        self.tf_session.run(tf.global_variables_initializer())

    def get_gradients(self, states_im, actions):
        return self.tf_session.run(
            self.critic_gradients,
            feed_dict={self.state_input: states_im, self.action_input: actions},
        )[0]

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        state_input = Input(shape=(IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS), name='Input_image')

        conv0 = Conv2D(FILTERS_CONV, KERNEL_CONV, padding='same', activation='relu', init='uniform', name='conv0')(state_input)
        av0 = MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding='same', name='av0')(conv0)

        # conv1 = Conv2D(20, KERNEL_CONV, padding='same', activation='relu', init='uniform', name='conv1')(av0)
        # av1 = MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding='same', name='av1')(conv1)

        flat = Flatten()(av0)

        state_h1 = Dense(hidden_units[1], activation="relu")(flat)
        state_h2 = Dense(hidden_units[2], activation="linear")(state_h1)

        action_input = Input(shape=[2])
        action_h1 = Dense(hidden_units[2], activation="linear")(action_input)

        #merged = concatenate([state_h2, action_h1])
        merged = add([state_h2, action_h1])
        merged_h1 = Dense(hidden_units[2], activation="relu")(merged)

        output_layer = Dense(1, activation="linear")(merged_h1)
        model = Model(input=[state_input, action_input], output=output_layer)

        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        tf.keras.utils.plot_model(model, to_file=image_network + 'critic_model_CNN.png',
                                  show_shapes=True, show_layer_names=True, rankdir='TB')




        # state_input = Input(shape=(IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS), name='Input_image')
        #
        # conv0 = Conv2D(32, (3, 3), padding='same', activation='relu', init='uniform', name='conv0')(state_input)
        # av0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='av0')(conv0)
        #
        # conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', init='uniform', name='conv1')(av0)
        # av1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='av1')(conv1)
        #
        # conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', init='uniform', name='conv2')(av1)
        # av2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='av2')(conv2)
        #
        # flat = Flatten(name='flat')(av2)
        # dense1 = Dense(128, activation='relu', name='dense1')(flat)
        #
        # action_input = Input(shape=[2], name='Input_action')
        # dense2 = Dense(128, activation='relu', name='dense2')(action_input)
        #
        # merged = add([dense1, dense2], name='merged')
        # merged_h1 = Dense(128, activation="relu", name='merged_h1')(merged)
        #
        # output_layer = Dense(1, activation="tanh", name='output_layer')(merged_h1)
        # model = Model(input=[state_input, action_input], output=output_layer)
        #
        # model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        # tf.keras.utils.plot_model(model,
        #                           to_file='NETWORKS/critic_model_image.png',
        #                           show_shapes=True,
        #                           show_layer_names=True, rankdir='TB')



        return model, state_input, action_input
