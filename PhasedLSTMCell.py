from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops


def random_exp_initializer(minval=0, maxval=None, seed=None,
                               dtype=dtypes.float32):
    """Returns an initializer that generates tensors with an exponential distribution.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type.

    Returns:
      An initializer that generates tensors with an exponential distribution.
    """

    def _initializer(shape, dtype=dtype, partition_info=None):
        return tf.exp(random_ops.random_uniform(shape, minval, maxval, dtype, seed=seed))

    return _initializer

# Here we need to register the gradient for the mod operation
@ops.RegisterGradient("Mod")
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    y_grad = tf.reduce_mean(-(x // y) * gz, reduction_indices=[0], keep_dims=True)
    return x_grad, y_grad


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name + "/concat"
    concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
    for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value

    concat_variable = array_ops.concat(0, sharded_variable, name=concat_name)
    ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                          concat_variable)
    return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(vs.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                      dtype=dtype))
    return shards


class PhasedLSTMCell(RNNCell):
    """Phased Long short-term memory unit (PLSTM) recurrent network cell.

    The default non-peephole implementation is based on:

      http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    The peephole implementation is based on:

      https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The Kronos gate implementation is  based on:

      https://arxiv.org/abs/1610.09513

    Daniel Neil, Michael Pfeiffer, Shih-Chii Liu.
    "Phased LSTM: Accelerating Recurrent Network
     Training for Long or Event-based Sequences"


    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, num_units, input_size=None,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=1, num_proj_shards=1,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=tanh, alpha=0.001, r_on_init=0.05, tau_init=6.,
                 manual_set=False,trainable=True):
        """Initialize the parameters for an PLSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          input_size: Deprecated and unused.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
          provided, then the projected values are clipped elementwise to within
          `[-proj_clip, proj_clip]`.
          num_unit_shards: How to split the weight matrix.  If >1, the weight
            matrix is stored across num_unit_shards.
          num_proj_shards: How to split the projection matrix.  If >1, the
            projection matrix is stored across num_proj_shards.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
          alpha: (optional) A Float value. Decay rate during the off period of the
            kronos gate.
          r_on_init: (optional) A Float value. Initial value for r_on
          tau_init: (optional) A Float value. Max value for the exponential
            initialization of tau
          manual_set: (optional) If True, tau_init is set as a constant value
            instead of being randomised (default behavioiur) and the phase variable
            s is set to zero. The kronos gate behaviour is hard on during r_on.
            This mimics the behaviour of the audio/video input layers of the Lip
            Reading experiment in the Phased LSTM paper. Default value: False.
          trainable: (optional) If False, the trainable parameter of variable tau,
            r_on and s are set to False such that learning is disabled on these
            parameters. Default value: True.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self.alpha = alpha
        self.r_on_init = r_on_init
        self.tau_init = tau_init

        self.manual_set = manual_set
        self.trainable = trainable

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".

        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            i_size = input_size.value - 1  # -1 to extract time
            times = array_ops.slice(inputs, [0, i_size], [-1, 1])
            filtered_inputs = array_ops.slice(inputs, [0, 0], [-1, i_size])

            # --------------------------------------- #
            # ------------- PHASED LSTM ------------- #
            # ---------------- BEGIN ---------------- #
            # --------------------------------------- #

            tau = vs.get_variable(
                "T", shape=[self._num_units],
                initializer=random_exp_initializer(0, self.tau_init) if not self.manual_set else init_ops.constant_initializer(self.tau_init),
                trainable=self.trainable,dtype=dtype)

            r_on = vs.get_variable(
                "R", shape=[self._num_units],
                initializer=init_ops.constant_initializer(self.r_on_init),
                trainable=self.trainable, dtype=dtype)

            s = vs.get_variable(
                "S", shape=[self._num_units],
                initializer=init_ops.random_uniform_initializer(0., tau.initialized_value()) if not self.manual_set else init_ops.constant_initializer(0.),
                trainable=self.trainable,dtype=dtype)
                # for backward compatibility (v < 0.12.0) use the following line instead of the above
                # initializer = init_ops.random_uniform_initializer(0., tau), dtype = dtype)

            tau_broadcast = tf.expand_dims(tau, dim=0)
            r_on_broadcast = tf.expand_dims(r_on, dim=0)
            s_broadcast = tf.expand_dims(s, dim=0)

            r_on_broadcast = tf.abs(r_on_broadcast)
            tau_broadcast = tf.abs(tau_broadcast)
            times = tf.tile(times, [1, self._num_units])

            # calculate kronos gate
            phi = tf.div(tf.mod(tf.mod(times - s_broadcast, tau_broadcast) + tau_broadcast, tau_broadcast),
                         tau_broadcast)
            is_up = tf.less(phi, (r_on_broadcast * 0.5))
            is_down = tf.logical_and(tf.less(phi, r_on_broadcast), tf.logical_not(is_up))

            # when manually setting, hard on over r_on, else as previous
            if self.manual_set:
                k = tf.select(tf.logical_or(is_up,is_down), tf.to_float(is_up), self.alpha * phi)
            else:
                k = tf.select(is_up, phi / (r_on_broadcast * 0.5),
                              tf.select(is_down, 2. - 2. * (phi / r_on_broadcast), self.alpha * phi))

            # --------------------------------------- #
            # ------------- PHASED LSTM ------------- #
            # ----------------- END ----------------- #
            # --------------------------------------- #

            concat_w = _get_concat_variable(
                "W", [i_size + num_proj, 4 * self._num_units],
                dtype, self._num_unit_shards)

            b = vs.get_variable(
                "B", shape=[4 * self._num_units],
                initializer=init_ops.zeros_initializer, dtype=dtype)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = array_ops.concat(1, [filtered_inputs, m_prev])
            lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
            i, j, f, o = array_ops.split(1, 4, lstm_matrix)

            # Diagonal connections
            if self._use_peepholes:
                w_f_diag = vs.get_variable(
                    "W_F_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = vs.get_variable(
                    "W_I_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = vs.get_variable(
                    "W_O_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                     sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
                # pylint: enable=invalid-unary-operand-type

            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                concat_w_proj = _get_concat_variable(
                    "W_P", [self._num_units, self._num_proj],
                    dtype, self._num_proj_shards)

                m = tf.math_ops.matmul(m, concat_w_proj)
                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type

            # APPLY KRONOS GATE
            c = k * c + (1. - k) * c_prev
            m = k * m + (1. - k) * m_prev
            # END KRONOS GATE

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple
                     else array_ops.concat(1, [c, m]))
        return m, new_state


def multiPLSTM(input, batch_size, lens, n_layers, units_p_layer, n_input, initial_states=None):
    """
    Function to build multilayer PLSTM
    :param input: 3D tensor, where the time input is appended and represents the last feature of the tensor
    :param batch_size: integer, batch size
    :param lens: 2D tensor, length of the sequences in the batch (for synamic rnn use)
    :param n_layers: integer, number of layers
    :param units_p_layer: integer, number of units per layer
    :param n_input: integer, number of features in the input (without time feature)
    :param initial_states: list of tuples of initial states
    :return: 3D tensor, output of the multilayer PLSTM
    """
    if initial_states is None:
        initial_states = [tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, units_p_layer], tf.float32),
                                                        tf.zeros([batch_size, units_p_layer], tf.float32))
                          for _ in range(n_layers)]

    assert (len(initial_states) == n_layers)
    times = tf.slice(input, [0, 0, n_input], [-1, -1, 1])
    newX = tf.slice(input, [0, 0, 0], [-1, -1, n_input])

    for k in range(n_layers):
        newX = tf.concat(2, [newX, times])
        with tf.variable_scope("{}".format(k)):
            cell = PhasedLSTMCell(units_p_layer, use_peepholes=True,
                                  state_is_tuple=True)
            outputs, initial_states[k] = tf.nn.dynamic_rnn(cell, newX, dtype=tf.float32,
                                                           sequence_length=lens,
                                                           initial_state=initial_states[k])
            newX = outputs
    return newX, initial_states
