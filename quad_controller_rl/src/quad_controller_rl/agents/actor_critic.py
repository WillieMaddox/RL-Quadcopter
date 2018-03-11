import tensorflow as tf
# from keras.utils import plot_model
from keras.layers import Add
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import Ones, Zeros
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class LayerNorm1D(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=Zeros(),
                                    trainable=True,)

        super().build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class BaseModel(object):
    def __init__(self, name):
        self.name = name
        self.u1 = 32
        self.u2 = 64

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'layer_norm' not in var.name]


class Actor(BaseModel):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, name='actor', layer_norm=True, reuse=False):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        super(Actor, self).__init__(name)
        self.layer_norm = layer_norm
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model(reuse=reuse)

    def build_model(self, reuse):
        """Build an actor (policy) network that maps states -> actions."""

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # Define input layer (states)
            states = Input(shape=(self.state_size,), name='states')
            # Add hidden layers
            net = Dense(units=self.u1)(states)
            if self.layer_norm:
                net = LayerNorm1D()(net)
            # net = Activation('relu')(net)
            net = LeakyReLU(alpha=0.01)(net)
            net = Dense(units=self.u2)(net)
            if self.layer_norm:
                net = LayerNorm1D()(net)
            # net = Activation('relu')(net)
            net = LeakyReLU(alpha=0.01)(net)
            net = Dense(units=self.u1)(net)
            if self.layer_norm:
                net = LayerNorm1D()(net)
            # net = Activation('relu')(net)
            net = LeakyReLU(alpha=0.01)(net)

            # Try different layer sizes, activations, add batch normalization, regularizers, etc.

            # Add final output layer with sigmoid activation
            raw_actions = Dense(units=self.action_size, name='raw_actions')(net)
            raw_actions = Activation('sigmoid')(raw_actions)

            # # Left/Right --> [-1, 1]
            # fx = Dense(units=1, name='fx')(net)
            # fx = Activation('tanh')(fx)
            # # Forward/Backward --> [-1, 1]
            # fy = Dense(units=1, name='fy')(net)
            # fy = Activation('tanh')(fy)
            # # No thrust down, only thrust up --> [0, 1]
            # fz = Dense(units=1, name='fz')(net)
            # fz = Activation('sigmoid')(fz)
            # raw_actions = Concatenate()([fx, fy, fz])

            # Scale [0, 1] output for each action dimension to proper range
            actions = Lambda(lambda x: x * self.action_range + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=(self.action_size,), name='action_gradients')
        loss = -K.mean(action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = Adam(lr=0.0001)
        updates_op = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)

        self.set_action_gradients = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[loss],
            updates=updates_op)

        # self.model.summary()
        # plot_model(self.model, to_file=self.name + '.png')


class Critic(BaseModel):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, name='critic', layer_norm=True, reuse=False):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__(name)
        self.layer_norm = layer_norm
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model(reuse)

    def build_model(self, reuse):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # Define input layers and add hidden layer(s) for state pathway
            states = Input(shape=(self.state_size,), name='states')

            net_states = Dense(units=self.u1)(states)
            if self.layer_norm:
                net_states = LayerNorm1D()(net_states)
            # net_states = Activation('relu')(net_states)
            net_states = LeakyReLU(alpha=0.01)(net_states)

            net_states = Dense(units=self.u2)(net_states)
            if self.layer_norm:
                net_states = LayerNorm1D()(net_states)
            # net_states = Activation('relu')(net_states)
            net_states = LeakyReLU(alpha=0.01)(net_states)

            # Define input layers and add hidden layer(s) for action pathway
            actions = Input(shape=(self.action_size,), name='actions')

            net_actions = Dense(units=self.u1)(actions)
            if self.layer_norm:
                net_actions = LayerNorm1D()(net_actions)
            # net_actions = Activation('relu')(net_actions)
            net_actions = LeakyReLU(alpha=0.01)(net_actions)

            net_actions = Dense(units=self.u2)(net_actions)
            if self.layer_norm:
                net_actions = LayerNorm1D()(net_actions)
            # net_actions = Activation('relu')(net_actions)
            net_actions = LeakyReLU(alpha=0.01)(net_actions)

            # Try different layer sizes, activations, add batch normalization, regularizers, etc.

            # Combine state and action pathways
            net = Concatenate()([net_states, net_actions])
            # net = Add()([net_states, net_actions])

            # Add more layers to the combined network if needed
            net = Dense(units=self.state_size+self.action_size)(net)
            # if self.layer_norm:
            #     net = LayerNorm1D()(net)
            # net = Activation('linear')(net)

            # Add final output layer to produce action values (Q values)
            q_values = Dense(units=1, name='q_values')(net)

        self.model = Model(inputs=[states, actions], outputs=q_values)
        self.model.compile(optimizer=Adam(), loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

        # self.model.summary()
        # plot_model(self.model, to_file=self.name + '.png')

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
