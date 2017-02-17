from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import _RNNCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest


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
@ops.RegisterGradient("FloorMod")
def _mod_grad(op, grad):
    x, y = op.inputs
    gz = grad
    x_grad = gz
    y_grad = tf.reduce_mean(-(x // y) * gz, axis=[0], keep_dims=True)
    return x_grad, y_grad


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: (optional) Variable scope to create parameters in.
  
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]
    
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    
    dtype = [a.dtype for a in args][0]
    
    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)


class PhasedLSTMCell(_RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
  
    The default non-peephole implementation is based on:
  
      http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
  
    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  
    The peephole implementation is based on:
  
      https://research.google.com/pubs/archive/43905.pdf
  
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.
  
    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """
    
    def __init__(self, num_units, input_size=None,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=tanh, alpha=0.001, r_on_init=0.05, tau_init=6.,
                 manual_set=False, trainable=True):
        """Initialize the parameters for an LSTM cell.
    
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
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)
        
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
          scope: VariableScope for the created subgraph; defaults to "lstm_cell".
    
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
        with vs.variable_scope(scope or "lstm_cell",
                               initializer=self._initializer) as unit_scope:
            if self._num_unit_shards is not None:
                unit_scope.set_partitioner(
                    partitioned_variables.fixed_size_partitioner(
                        self._num_unit_shards))
            
            i_size = input_size.value - 1  # -1 to extract time
            times = array_ops.slice(inputs, [0, i_size], [-1, 1])
            filtered_inputs = array_ops.slice(inputs, [0, 0], [-1, i_size])
            
            # --------------------------------------- #
            # ------------- PHASED LSTM ------------- #
            # ---------------- BEGIN ---------------- #
            # --------------------------------------- #
            
            tau = vs.get_variable(
                "T", shape=[self._num_units],
                initializer=random_exp_initializer(0,
                                                   self.tau_init) if not self.manual_set else init_ops.constant_initializer(
                    self.tau_init),
                trainable=self.trainable, dtype=dtype)
            
            r_on = vs.get_variable(
                "R", shape=[self._num_units],
                initializer=init_ops.constant_initializer(self.r_on_init),
                trainable=self.trainable, dtype=dtype)
            
            s = vs.get_variable(
                "S", shape=[self._num_units],
                initializer=init_ops.random_uniform_initializer(0.,
                                                                tau.initialized_value()) if not self.manual_set else init_ops.constant_initializer(
                    0.),
                trainable=self.trainable, dtype=dtype)
            # for backward compatibility (v < 0.12.0) use the following line instead of the above
            # initializer = init_ops.random_uniform_initializer(0., tau), dtype = dtype)
            
            tau_broadcast = tf.expand_dims(tau, axis=0)
            r_on_broadcast = tf.expand_dims(r_on, axis=0)
            s_broadcast = tf.expand_dims(s, axis=0)
            
            r_on_broadcast = tf.abs(r_on_broadcast)
            tau_broadcast = tf.abs(tau_broadcast)
            times = tf.tile(times, [1, self._num_units])
            
            # calculate kronos gate
            phi = tf.div(tf.mod(tf.mod(times - s_broadcast, tau_broadcast) + tau_broadcast, tau_broadcast),
                         tau_broadcast)
            is_up = tf.less(phi, (r_on_broadcast * 0.5))
            is_down = tf.logical_and(tf.less(phi, r_on_broadcast), tf.logical_not(is_up))
            
            k = tf.where(is_up, phi / (r_on_broadcast * 0.5),
                         tf.where(is_down, 2. - 2. * (phi / r_on_broadcast), self.alpha * phi))
            
            # --------------------------------------- #
            # ------------- PHASED LSTM ------------- #
            # ----------------- END ----------------- #
            # --------------------------------------- #
            
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            lstm_matrix = _linear([filtered_inputs, m_prev], 4 * self._num_units, bias=True,
                                  scope=scope)
            i, j, f, o = array_ops.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)
            
            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope) as projection_scope:
                    if self._num_unit_shards is not None:
                        projection_scope.set_partitioner(None)
                    w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)
            
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
                with vs.variable_scope("projection") as proj_scope:
                    if self._num_proj_shards is not None:
                        proj_scope.set_partitioner(
                            partitioned_variables.fixed_size_partitioner(
                                self._num_proj_shards))
                    m = _linear(m, self._num_proj, bias=False, scope=scope)
                
                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type
            
            # APPLY KRONOS GATE
            c = k * c + (1. - k) * c_prev
            m = k * m + (1. - k) * m_prev
            # END KRONOS GATE
        
        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state


def multiPLSTM(cells, inputs, lens, n_input, initial_states):
    """
    Function to build multilayer PLSTM
    :param cells:
    :param inputs:
    :param lens: 2D tensor, length of the sequences in the batch (for synamic rnn use)
    :param n_input: integer, number of features in the input (without time feature)
    :param initial_states: list of tuples of initial states
    :return: 3D tensor, output of the multilayer PLSTM
    """
    
    assert (len(initial_states) == len(cells))
    times = tf.slice(inputs, [0, 0, n_input], [-1, -1, 1])
    new_x = tf.slice(inputs, [0, 0, 0], [-1, -1, n_input])
    
    for k, cell, initial_state in zip(range(len(cells)), cells, initial_states):
        new_x = tf.concat(axis=2, values=[new_x, times])
        with tf.variable_scope("{}".format(k)):
            outputs, initial_states[k] = tf.nn.dynamic_rnn(cell, new_x, dtype=tf.float32,
                                                       sequence_length=lens,
                                                       initial_state=initial_state)
            new_x = outputs
    return new_x
