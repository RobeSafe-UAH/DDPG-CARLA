import keras.backend as keras_backend
import tensorflow as tf
from keras.layers import Dense, Input, merge, add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from carla_config import hidden_units, image_network


class CriticNetwork:
    def __init__(self, tf_session, state_size, action_size=1, tau=0.001, lr=0.001):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr

        keras_backend.set_session(tf_session)

        self.model, self.state_input, self.action_input = self.generate_model()

        self.target_model, _, _ = self.generate_model()

        self.critic_gradients = tf.gradients(self.model.output, self.action_input)
        self.tf_session.run(tf.global_variables_initializer())

    def get_gradients(self, states, actions):
        return self.tf_session.run(
            self.critic_gradients,
            feed_dict={self.state_input: states, self.action_input: actions},
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
        state_input = Input(shape=[self.state_size])
        state_h1 = Dense(hidden_units[0], activation="relu")(state_input)
        state_h2 = Dense(hidden_units[1], activation="linear")(state_h1)

        action_input = Input(shape=[self.action_size])
        action_h1 = Dense(hidden_units[1], activation="linear")(action_input)

        merged = add([state_h2, action_h1])
        merged_h1 = Dense(hidden_units[1], activation="relu")(merged)

        output_layer = Dense(1, activation="linear")(merged_h1)
        model = Model(input=[state_input, action_input], output=output_layer)




        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        tf.keras.utils.plot_model(model, to_file=image_network + 'critic_model_WP_Carla.png',
                                  show_shapes=True, show_layer_names=True, rankdir='TB')
        return model, state_input, action_input
