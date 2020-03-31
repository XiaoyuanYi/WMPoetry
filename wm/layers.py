from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical

class Embedding(object):
    """
    Embedding layer
    """
    def __init__(self,
               vocab_size, emb_size,
               init_emb = None, name = 'embedding',
               trainable=True):

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.name = name
        self.trainable = trainable
        self.reuse = None
        self.trainable_weights = None
        self.init_emb = init_emb

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            if self.init_emb is not None:
                initializer = tf.constant_initializer(self.init_emb)
                if not self.reuse:
                    print ("Initialize embedding with pre-trained vectors.")
            else:
                initializer = tf.truncated_normal_initializer(stddev=1e-4)
                if not self.reuse:
                    print ("Initialize embedding with normal distribution.")

            word_emb = tf.get_variable('word_emb', [self.vocab_size, self.emb_size], 
                dtype=tf.float32, initializer=initializer, trainable= self.trainable)

            embedded = tf.nn.embedding_lookup(word_emb, x)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return embedded

class BidirEncoder(object):
    """
    Bidirectional Encoder
    Exposes variables in `trainable_weights` property.
    """
    def __init__(self, cell_size, keep_prob=1.0, name='bidirenc'):
        self.cell_size = cell_size
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell_fw = tf.nn.rnn_cell.GRUCell(
                self.cell_size, reuse=tf.get_variable_scope().reuse)
            cell_bw = tf.nn.rnn_cell.GRUCell(
                self.cell_size, reuse=tf.get_variable_scope().reuse)

            cell_fw =  tf.nn.rnn_cell.DropoutWrapper(cell_fw, 
                output_keep_prob=self.keep_prob, 
                input_keep_prob = self.keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, 
                output_keep_prob=self.keep_prob, 
                input_keep_prob = self.keep_prob)

            outs , state_fw, state_bw  = tf.nn.static_bidirectional_rnn(
                    cell_fw, cell_bw, x, dtype=tf.float32)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return outs, state_fw, state_bw

class Decoder(object):
    '''
    Decoder with an output projection layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, cell_size, vocab_size, keep_prob=1.0, name='dec'):
        self.cell_size = cell_size
        self.keep_prob = keep_prob
        self.vocab_size = vocab_size
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, state, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.nn.rnn_cell.GRUCell(
                self.cell_size, reuse=tf.get_variable_scope().reuse)

            cell =  tf.nn.rnn_cell.DropoutWrapper(cell, 
                output_keep_prob=self.keep_prob, 
                input_keep_prob = self.keep_prob)

            cell_out, next_state = cell(x, state)
            out = linear(cell_out, self.vocab_size, 
                bias=True, scope="out_proj")

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return out, next_state


class AttentionLayer(object):
    '''
    The attention layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, name='attention'):
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, attentions, query, attn_mask):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            attn_length = attentions.get_shape()[1].value  # the length of a input sentence
            attn_size = attentions.get_shape()[2].value  # the size of each slot
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            # Remember we use bidirectional RNN
            hidden = array_ops.reshape(attentions, [-1, attn_length, 1, attn_size]) 

            k = tf.get_variable(shape=[1, 1, attn_size, attn_size], name="AttnW")
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable(shape=[attn_size], name="AttnV")

            y = linear(query, attn_size, bias=True, scope='attn_query_trans')
            y = array_ops.reshape(y, [-1, 1, 1, attn_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum( v * math_ops.tanh(hidden_features + y), [2, 3]) 
            a = nn_ops.softmax(s + attn_mask)
            # Now calculate the attention-weighted vector d.
            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            d = array_ops.reshape(d, [-1, attn_size])  #remember this size

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return d, a

class AttnWriteLayer(object):
    #The attention layer for memory writing
    #Exposes variables in `trainable_weights` property.
    def __init__(self, mem_slots, mem_size, name='attn_write'):
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, his_mem, enc_states, global_trace, write_mask, null_mem, gama):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:

            for i, mstate in enumerate(enc_states):
                if i > 0:
                    vs.reuse_variables()
                # Concatenate history memory with the null slot
                mem = array_ops.concat([his_mem, tf.identity(null_mem)], axis=1)

                hidden = array_ops.reshape(mem, [-1, self.mem_slots+1, 1, self.mem_size]) 
                k = tf.get_variable("AttnW", [1, 1, self.mem_size, self.mem_size])
                mem_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                v = tf.get_variable("AttnV", [self.mem_size])
                bias = tf.get_variable("Attnb", [self.mem_slots+1])

                y = linear([mstate, global_trace], self.mem_size, True, scope="query_trans")
                y = array_ops.reshape(y, [-1, 1, 1, self.mem_size])
                s = math_ops.reduce_sum(v * math_ops.tanh(mem_features + y), [2, 3]) + bias
                
                # The random_mask shows if a slot is empty, 1 empty, 0 not empty.
                random_mask = 1.0 - tf.sign(math_ops.reduce_sum(tf.abs(mem), axis=2))

                # We use sampling to produce random bias for empty slots
                q = RelaxedOneHotCategorical(gama, logits=s)
                sampled_q = q.sample()
                annealing_q = nn_ops.softmax(s)

                # We manually give the empty slots higher weights
                walign = random_mask * sampled_q * 5.0 + (1-random_mask) * annealing_q

                # normalization
                norm = math_ops.reduce_sum(walign, axis=1)
                norm = tf.expand_dims(norm, axis=1)
                norm_align = math_ops.divide(walign, norm)

                select_q = tf.expand_dims(math_ops.argmax(norm_align, -1), axis=1)
                select_q = tf.cast(select_q, norm_align.dtype)
                indices0 = tf.stop_gradient(select_q - norm_align) + norm_align

                indices = math_ops.reduce_mean(indices0, axis=-1)
                q = tf.one_hot(tf.cast(indices, tf.int32), self.mem_slots+1)
                q = tf.cast(q, tf.float32)
                q = tf.stop_gradient(q-indices0) + indices0

                masked_align = tf.multiply(write_mask[i], q)
                float_mask = masked_align[:, 0:self.mem_slots]
                final_mask = tf.expand_dims(float_mask, axis=2)

                w_states = tf.tile(mstate, [1, self.mem_slots])
                w_states = array_ops.reshape(w_states, [-1, self.mem_slots, self.mem_size])

                his_mem = (1.0 - final_mask) * his_mem + final_mask * w_states


            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return his_mem
         

class GTraceLayer(object):
    '''
    Global Trace layer. This layer uses convolution to 
        updata the global trace, which is more powerful
        than the simple average of hidden states in our old version.
    '''
    def __init__(self, filter_size, filter_num, name='global_trace'):
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, global_h, ori_states):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            # ori_states: [batch_size, enc_len, mem_size]
            states = tf.expand_dims(ori_states, axis=-1)
            m = states.get_shape()[2].value
            
            k = tf.get_variable("conv", [self.filter_size, m, 1, self.filter_num])
            bias = tf.get_variable("bias", shape=[self.filter_num], dtype=tf.float32)

            hidden_features = nn_ops.conv2d(states, k, [1, 1, 1, 1], "VALID")
            features = tf.tanh(tf.nn.bias_add(hidden_features, bias))

            pool_size = features.get_shape()[1]
            pool_stride = 1
                
            features = tf.nn.avg_pool(features, [1, pool_size, 1, 1],
                [1, pool_stride, 1, 1], padding='VALID')
            features = tf.squeeze(features, [1,2])

            new_global_h = tf.tanh(linear([features, global_h], global_h.get_shape()[1], 
                    bias=True, scope="update_history" ))

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
                
        return new_global_h


class MLPLayer(object):
    '''
    MLP layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, out_sizes, activs=None, keep_prob=1.0, trainable=True, name='mlp'):
        
        self.out_sizes = out_sizes
        if activs is None:
            activs = ['tanh'] * len(out_sizes)
        self.activs = activs
        self.keep_prob = keep_prob
        self.trainable = trainable
        self.name = name
        self.reuse = None
        self.trainable_weights = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:

            out = x
            layers_num = len(self.out_sizes)
            for i, (out_size, activ) in enumerate(zip(self.out_sizes, self.activs)):
                out = linear(out, out_size, bias=True, scope="mlp"+str(i), trainable=self.trainable)
                assert activ == 'tanh' or activ == 'relu' or activ == 'leak_relu' or activ is None
                if activ == 'tanh':
                    out =  tf.tanh(out)
                elif activ == 'relu':
                    out =  tf.nn.relu(out)
                elif activ == 'leak_relu':
                    LeakyReLU = tf.keras.layers.LeakyReLU(0.2)
                    out = LeakyReLU(out)

                if layers_num > 1 and i < layers_num-1:
                    out = tf.nn.dropout(out, self.keep_prob)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return out


# ----------------------------------------
# Tool functions
class SoftmaxLoss(object):
    '''Softmax xentropy'''
    def __init__(self, name):
        self.name = name

    def __call__(self, logits, targets, weights,
        average_across_timesteps=True, average_across_batch=True,
        softmax_loss_function=None, name=None):
        
        with ops.name_scope(self.name):
            cost = math_ops.reduce_sum(sequence_loss_by_example(
                logits, targets, weights,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))

            if average_across_batch:
                batch_size = array_ops.shape(targets[0])[0]
                return cost / math_ops.cast(batch_size, cost.dtype)
            else:
                return cost

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
            "%d, %d, %d." % (len(logits), len(weights), len(targets)))

    with ops.name_scope("sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size

    return log_perps


def linear(inputs, output_size, bias=True, concat=True, trainable=True, dtype=None, scope=None):
    """
    Linear layer. The code of this funciton is from the THUMT project
    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param dtype: an instance of tf.DType, the default value is ``tf.float32''
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)
            shape = [input_size, output_size]
            matrix = tf.get_variable("Matrix", shape, dtype=dtype, trainable=trainable)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype, trainable=trainable)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("Bias", shape, dtype=dtype, trainable=trainable)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output