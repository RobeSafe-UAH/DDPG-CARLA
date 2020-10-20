import keras.backend as keras_backend
import tensorflow as tf
from keras.initializers import normal
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, Activation, Flatten, Concatenate, MaxPooling2D
from keras.models import Model
from carla_config import IM_HEIGHT_CNN, IM_WIDTH_CNN, IM_LAYERS, hidden_units, FILTERS_CONV, KERNEL_CONV, POOL_SIZE, POOL_STRIDES, image_network

class ActorNetwork_CNN:
    def __init__(self, tf_session, tau=0.001, lr=0.0001):
        self.tf_session = tf_session
        self.action_size = 2
        self.tau = tau
        self.lr = lr


        keras_backend.set_session(tf_session)

        self.model, self.model_states = self.generate_model()
        model_weights = self.model.trainable_weights

        self.target_model, _ = self.generate_model()

        # Generate tensors to hold the gradients for Policy Gradient update
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, model_weights)

        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())

    def train(self, states_im, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_states: states_im,
                self.action_gradients: action_gradients,
            },
        )

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        input_layer = Input(shape=(IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS))

        conv0 = Conv2D(FILTERS_CONV, KERNEL_CONV, padding='same', activation='relu', init='uniform', name='conv0')(input_layer)
        av0 = MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding='same', name='av0')(conv0)

        # conv1 = Conv2D(20, KERNEL_CONV, padding='same', activation='relu', init='uniform', name='conv1')(av0)
        # av1 = MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding='same', name='av1')(conv1)

        flat = Flatten()(av0)

        h0 = Dense(hidden_units[1], activation="relu")(flat)
        h1 = Dense(hidden_units[2], activation="relu")(h0)
        output_layer = Dense(self.action_size, activation="tanh")(h1)
        model = Model(input=input_layer, output=output_layer)
        tf.keras.utils.plot_model(model,
                                  to_file=image_network + 'actor_model_CNN.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')



        # state_input = Input(shape=(IM_WIDTH_CNN, IM_HEIGHT_CNN, IM_LAYERS))
        #
        # conv0 = Conv2D(32, (3, 3), padding='same', activation='relu', init='uniform')(state_input)
        # av0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv0)
        #
        # conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', init='uniform')(av0)
        # av1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        #
        # conv2 = Conv2D(128, (3, 3), padding='same', activation='relu', init='uniform')(av1)
        # av2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        #
        # flat = Flatten()(av2)
        #
        # merged_h1 = Dense(128, activation="relu")(flat)
        #
        # output_layer = Dense(self.action_size, activation="tanh")(merged_h1)
        # model = Model(input=state_input, output=output_layer)
        #
        # tf.keras.utils.plot_model(model,
        #                           to_file='NETWORKS/actor_model_image.png',
        #                           show_shapes=True,
        #                           show_layer_names=True, rankdir='TB')



        return model, input_layer
