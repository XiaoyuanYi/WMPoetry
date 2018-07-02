from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from nn import linear
from functions import sequence_loss, flatten_query

class PoemModel(object):
    def __init__(self, hps, init_emb=None):
        # Create the model
        self.hps = hps
        self.enc_len = hps.bucket[0]
        self.dec_len = hps.bucket[1]
        self.mode = self.hps.mode
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)

        self.zeros = np.zeros((self.hps.batch_size, self.hps.his_mem_slots), dtype=np.float32)
        self.gama = tf.constant(20.0)

        self.learning_rate = tf.Variable(float(self.hps.learning_rate), trainable=False)
        self.learning_rate_decay_op = \
            self.learning_rate.assign(self.learning_rate * self.hps.decay_rate)

        self.b_size = self.hps.batch_size if self.mode == 'train' else 1
        
        # Random bias for memory writing
        init_val = []
        v = 1.0
        for i in xrange(0, self.hps.his_mem_slots):
            init_val.append(v)
            v /= 5.0

        bias_val = []
        for i in xrange(0, self.b_size):
            if self.b_size > 1:
                random.shuffle(init_val)
            bias_val.append(np.array(init_val+[0.0]))
        bias_val = np.array(bias_val)
        print ("Shape of random bias: %s" % (str(np.shape(bias_val))))
        self.random_bias = tf.constant(bias_val, dtype=tf.float32)

        # The null slot
        null_mem = np.zeros([self.b_size, self.hps.his_mem_size], dtype=np.float32) - 1e-2
        self.null_mem = np.expand_dims(null_mem, 1)

        # Build the graph
        self.__build_placeholders()

        # Build word embedding
        with tf.variable_scope('word_embedding'), tf.device('/cpu:0'):
            if init_emb is not None:
                print ("Initialize embedding with pre-trained word2vec.")
                initializer = tf.constant_initializer(init_emb)
            else:
                print ("Initialize embedding with normal distribution.")
                initializer = tf.truncated_normal_initializer(stddev=1e-4)
            word_emb = tf.get_variable('word_emb', [self.hps.vocab_size, self.hps.emb_size],
                dtype=tf.float32, initializer=initializer, trainable= True)

        self.emb_enc_inps = [ [tf.nn.embedding_lookup(word_emb, x) for x in enc_inp] for enc_inp in self.enc_inps]
        self.emb_dec_inps = [ [tf.nn.embedding_lookup(word_emb, x) for x in dec_inp] for dec_inp in self.dec_inps]
        self.emb_key_inps = [ [tf.nn.embedding_lookup(word_emb, x) for x in self.key_inps[i] ] 
            for i in xrange(0, self.hps.key_slots)]

        # Build genre embedding
        with tf.variable_scope('ph_embedding'), tf.device('/cpu:0'):
            # NOTE: we set fixed 36 phonology categories
            ph_emb = tf.get_variable('ph_emb', [36, self.hps.ph_emb_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_ph_inps = [ [tf.nn.embedding_lookup(ph_emb, x) for x in ph_inp] for ph_inp in self.ph_inps]

        with tf.variable_scope('len_embedding'), tf.device('/cpu:0'):
            len_emb = tf.get_variable('len_emb', [self.dec_len+1, self.hps.len_emb_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_len_inps = [ [tf.nn.embedding_lookup(len_emb, x) for x in len_inp] for len_inp in self.len_inps]

        # Concatenate phonology embedding and length embedding to form the genre embedding
        self.emb_genre = [[] for x in xrange(self.hps.sens_num)]
        for step in xrange(0, self.hps.sens_num):
            for i in xrange(self.dec_len+1):
                self.emb_genre[step].append(array_ops.concat([emb_ph_inps[step][i], emb_len_inps[step][i]], 1) )

        # Cells for the encoder
        enc_cell_fw  = tf.nn.rnn_cell.GRUCell(self.hps.hidden_size)
        enc_cell_bw = tf.nn.rnn_cell.GRUCell(self.hps.hidden_size)

        self.enc_cell_fw =  tf.nn.rnn_cell.DropoutWrapper(enc_cell_fw,  
            output_keep_prob=self.keep_prob, input_keep_prob = self.keep_prob)
        self.enc_cell_bw = tf.nn.rnn_cell.DropoutWrapper(enc_cell_bw, 
            output_keep_prob=self.keep_prob, input_keep_prob = self.keep_prob)

        if self.hps.mode == 'train':
            self.build_train_graph()
        else:
            self.build_gen_graph()

        # saver
        self.saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)

    def __build_placeholders(self):
        sens_num = self.hps.sens_num
        self.enc_inps = [[] for x in xrange(sens_num)]
        self.dec_inps = [[] for x in xrange(sens_num)]
        self.enc_mask  = [[] for x in xrange(sens_num)]
        self.ph_inps = [[] for x in xrange(sens_num)]
        self.len_inps = [[] for x in xrange(sens_num)]
        self.write_masks = [[] for x in xrange(sens_num)]

        for step in xrange(0, sens_num):
            for i in xrange(self.enc_len):
                self.enc_inps[step].append(tf.placeholder(tf.int32, shape=[None], name="enc{0}_{1}".format(step, i)))
                self.write_masks[step].append(tf.placeholder(tf.float32, shape=[None, 1], name="write_masks{0}_{1}".format(step, i)))
            for i in xrange(self.dec_len + 1):
                self.dec_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="dec{0}_{1}".format(step, i)))
                self.len_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="len_inps{0}_{1}".format(step, i)))
                self.ph_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="ph_inps{0}_{1}".format(step, i)))

            self.enc_mask[step] = tf.placeholder(tf.float32, shape=[None, self.enc_len, 1], name="enc_mask{0}".format(step))

        self.key_inps = [[] for x in xrange(self.hps.key_slots)]
        for i in xrange(self.hps.key_slots):
            # NOTE: We set that each keyword must consist of no more than 2 characters
            for j in xrange(2):
                self.key_inps[i].append(tf.placeholder(tf.int32, shape=[None], name="key{0}_{1}".format(i, j)))
        self.key_mask = tf.placeholder(tf.float32, shape=[None, self.hps.key_slots, 1], name="key_mask")

        self.trg_weights = [[] for x in xrange(sens_num)]
        for step in xrange(0, sens_num):
            for i in xrange(self.dec_len + 1):
                self.trg_weights[step].append(tf.placeholder(tf.float32, shape=[None], name="weight{0}_{1}".format(step, i)))

        self.targets = [[] for x in xrange(sens_num)]
        for step in xrange(0, sens_num):
            self.targets[step] = [self.dec_inps[step][i + 1] for i in xrange(len(self.dec_inps[step]) - 1)]
        
        # For beam search
        self.beam_key_align = tf.placeholder(tf.float32, 
            shape=[None, self.hps.key_slots], name="beam_key_align")
        self.beam_attn_states = tf.placeholder(tf.float32, 
            shape=[None, self.enc_len, self.hps.hidden_size*2], name="beam_attn_states")
        self.beam_initial_state = tf.placeholder(tf.float32, 
            shape=[None, self.hps.hidden_size] , name="beam_initial_state")
        self.beam_key_states = tf.placeholder(tf.float32, 
            shape=[None, self.hps.key_slots, self.hps.hidden_size*2], name="beam_key_states")
        self.beam_global_trace = tf.placeholder(tf.float32, 
            shape=[None, self.hps.global_trace_size], name="beam_global_trace")
        
        self.beam_his_mem = tf.placeholder(tf.float32, 
            [None, self.hps.his_mem_slots, self.hps.his_mem_size], name="beam_his_mem")
        self.beam_his_mem_mask = tf.placeholder(tf.float32, 
            [None, self.hps.his_mem_slots], name="beam_his_mem_mask")
        self.beam_topic_trace = tf.placeholder(tf.float32, 
            [None, self.hps.topic_trace_size + self.hps.key_slots], name="beam_topic_trace")

        # For history memory write
        self.beam_mem_states = []
        for i in xrange(self.dec_len):
            self.beam_mem_states.append(tf.placeholder(tf.float32, 
                shape=[None, self.hps.hidden_size*2],name="beam_mem_states{0}".format(i)))

    def build_gen_graph(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device), tf.variable_scope("graph"):
            self.key_initial_state, self.key_states = self.__build_key_memory()
            self.enc_state, self.attn_states, _  = self.__build_encoder(0)
            
            enc_mask_unpack =  tf.unstack(self.enc_mask[0], axis = 2)[0]
            key_mask_unpack =  tf.unstack(self.key_mask, axis = 2)[0]
            total_mask = array_ops.concat([self.beam_his_mem_mask, key_mask_unpack, enc_mask_unpack], 1)
            total_attn_states = array_ops.concat([self.beam_his_mem, self.beam_key_states, self.beam_attn_states], 1)
            
            # normed_dec_outs, dec_outs, dec_states, attn_weights
            self.next_out, _, self.next_state, self.next_align = self.__build_decoder([self.emb_dec_inps[0][0]],
                total_attn_states, total_mask, self.beam_global_trace, self.beam_initial_state, [self.emb_genre[0][0]], self.beam_topic_trace)
                    
            self.new_topic_trace = self.__topic_trace_update(self.beam_topic_trace, self.beam_key_align, self.beam_key_states)
            self.new_global_trace = self.__global_trace_update(self.beam_global_trace, self.beam_attn_states)
            self.new_his_mem =  self.__write_memory(self.beam_his_mem, 
                self.beam_mem_states, self.beam_global_trace, 0)

    def build_train_graph(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):

            normed_outs_vec, loss_vec = self.__build_graph()

            params = tf.trainable_variables()
                
            regularizers = []
            #print(len(params))
            for param in params:
                #name = param.name
                #print (name)
                regularizers.append(tf.nn.l2_loss(param))
                
            regular_loss = math_ops.reduce_sum(regularizers)
            gen_loss = math_ops.reduce_mean(loss_vec) 
            total_loss = gen_loss + 1e-5 * regular_loss
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = tf.gradients(total_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.hps.max_gradient_norm)
                
            self.update = opt.apply_gradients( zip(clipped_gradients, params), global_step=self.global_step)
            self.gradients = clipped_gradients
            
            # fetch
            self.gen_loss = gen_loss
            self.outputs = normed_outs_vec

    def __build_graph(self):
        with tf.variable_scope("graph"):
            key_initial_state, key_states = self.__build_key_memory()
            global_trace = array_ops.zeros([self.hps.batch_size, 
                self.hps.global_trace_size], dtype=tf.float32)
            
            his_mem = array_ops.zeros([self.hps.batch_size, self.hps.his_mem_slots, 
                self.hps.his_mem_size], dtype=tf.float32)
            
            topic_trace = array_ops.zeros([self.hps.batch_size, 
                self.hps.topic_trace_size+self.hps.key_slots], dtype=tf.float32)

            his_mem_mask = tf.constant(self.zeros)
            
            normed_outs_vec, loss_vec = [], []
            
            for step in xrange(0, self.hps.sens_num):
                if step > 0:
                        key_initial_state = None
                        variable_scope.get_variable_scope().reuse_variables()
                normed_outs, loss, global_trace, his_mem, topic_trace, \
                attn_weights = self.__build_seq2seq(global_trace, key_initial_state, 
                    key_states, his_mem, his_mem_mask, topic_trace, step)

                if step >=1:
                    his_mem_sum = math_ops.reduce_sum(tf.abs(his_mem), axis=2)
                    his_mem_bool = tf.not_equal(his_mem_sum, 0)
                    his_mem_mask = tf.to_float(his_mem_bool)

                normed_outs_vec.append(normed_outs)
                loss_vec.append(tf.identity(loss))
            return normed_outs_vec, loss_vec

    def __build_seq2seq(self, global_trace, key_initial_state, key_states, 
        his_mem, his_mem_mask, topic_trace, step):
        enc_state, attn_states, enc_outs = self.__build_encoder(step)
        if not (key_initial_state is None):
            initial_state = key_initial_state
        else:
            initial_state = enc_state

        enc_mask_unpack =  tf.unstack(self.enc_mask[step], axis = 2)[0]
        key_mask_unpack =  tf.unstack(self.key_mask, axis = 2)[0]
        total_mask = array_ops.concat( [his_mem_mask, key_mask_unpack, enc_mask_unpack], 1)
        total_attn_states = array_ops.concat( [his_mem, key_states, attn_states], 1)

        normed_dec_outs, dec_outs, dec_states, attn_weights = self.__build_decoder(
            self.emb_dec_inps[step][ : self.dec_len], total_attn_states, total_mask,
            global_trace, initial_state, self.emb_genre[step], topic_trace)

        concat_aligns = array_ops.concat(attn_weights, 0)
        key_align = concat_aligns[:, :, self.hps.his_mem_slots:self.hps.his_mem_slots+self.hps.key_slots]
        key_align = math_ops.reduce_mean(key_align, axis=0)
        new_topic_trace = self.__topic_trace_update(topic_trace, key_align, key_states)

        new_his_mem = self.__write_memory(his_mem, enc_outs, global_trace, step)
        new_global_trace = self.__global_trace_update(global_trace, attn_states)
        loss = sequence_loss(dec_outs, self.targets[step][: self.dec_len], self.trg_weights[step][ : self.dec_len])
        return normed_dec_outs, loss, new_global_trace, new_his_mem, new_topic_trace, attn_weights

    def __build_key_memory(self):
        #print ("memory")
        key_states = []
        with variable_scope.variable_scope("EncoderRNN"):
            for i in xrange(0, self.hps.key_slots):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                (outputs , state_fw, state_bw)  = rnn.static_bidirectional_rnn(
                    self.enc_cell_fw, self.enc_cell_bw, self.emb_key_inps[i], dtype=tf.float32)
                key_state = array_ops.concat([state_fw, state_bw], 1)
                key_states.append(key_state)
        
        with variable_scope.variable_scope("key_memory"):
            key_states = [array_ops.reshape(e, [-1, 1, self.enc_cell_fw.output_size*2]) for e in key_states]
            key_states = array_ops.concat(key_states, 1)
            key_states = tf.multiply(self.key_mask, key_states)

            final_state = math_ops.reduce_mean(key_states, axis=1)
            final_state = linear(final_state, self.hps.hidden_size, True,  scope="key_initial")
            final_state = tf.tanh(final_state)

        return final_state, key_states
            
    def __build_encoder(self, step):

        with variable_scope.variable_scope("EncoderRNN", reuse=True):
            (outputs , enc_state_fw, enc_state_bw)  = rnn.static_bidirectional_rnn(
                    self.enc_cell_fw, self.enc_cell_bw, self.emb_enc_inps[step][:self.enc_len], dtype=tf.float32)

            enc_outs = outputs

        with variable_scope.variable_scope("seq2seq_Encoder"):
            enc_state =  enc_state_bw
            final_state = linear(enc_state, self.hps.hidden_size, True,  scope="enc_initial")
            final_state = tf.tanh(final_state)

            top_states = [array_ops.reshape(e, [-1, 1, self.enc_cell_fw.output_size*2]) for e in enc_outs]
            attention_states = array_ops.concat( top_states, 1)

            final_attn_states = tf.multiply(self.enc_mask[step], attention_states)

        return final_state, final_attn_states, enc_outs

    def __build_decoder(self, dec_inps, total_attn_states, total_mask, global_trace,
        initial_state, genre, topic_trace):
        with variable_scope.variable_scope("seq2seq_Decoder"):

            dec_cell = tf.nn.rnn_cell.GRUCell(self.hps.hidden_size)       
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=self.keep_prob,
                input_keep_prob = self.keep_prob)

            output_size = self.hps.vocab_size # num_decoder_symbols is vocabulary size
            if not total_attn_states.get_shape()[1:3].is_fully_defined():
                raise ValueError("Shape[1] and [2] of attn_states must be known: %s" % total_attn_states.get_shape())

            dec_outs, dec_states, attn_weights  = [], [], []
            normed_dec_outs = []

            # build attn_mask
            imask = 1.0 - total_mask
            attn_mask = tf.where(tf.equal(imask, 1), tf.ones_like(imask) * (-float('inf')), imask)

            calcu_attention = lambda query : self.__attention_calcu(total_attn_states, query, attn_mask, "calcu_attention")

            state = initial_state
            decoder_input_size = initial_state.get_shape()[1].value
            for i, inp in enumerate(dec_inps):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                input_size = inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError("Could not infer input size from input: %s" % inp.name)

                # Read memory
                attns, align = calcu_attention([flatten_query(state), global_trace, topic_trace])

                with variable_scope.variable_scope("input_merge"):
                    x = linear([inp, attns, genre[i], global_trace], decoder_input_size, True)

                # Run the GRU
                cell_out, state = dec_cell(x, state)

                with variable_scope.variable_scope("OutputProjection"):
                    output = linear(cell_out, output_size, True)

                dec_outs.append(tf.identity(output))
                normed_dec_outs.append(tf.identity(tf.nn.softmax(output)))
                attn_weights.append([align])
                dec_states.append(tf.identity(state))

            return  normed_dec_outs, dec_outs, dec_states, attn_weights
    

    def __attention_calcu(self, attentions, query, attn_mask, scope):
        with variable_scope.variable_scope(scope):
            attn_length = attentions.get_shape()[1].value  # the length of a input sentence
            attn_size = attentions.get_shape()[2].value  # hidden state size of encoder, that is 2*size
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            # Remember we use bidirectional RNN 
            hidden = array_ops.reshape(attentions, [-1, attn_length, 1, attn_size]) 
            # Size of query vectors for attention
            # query vector is decoder state
            attention_vec_size = attn_size

            k = variable_scope.get_variable("AttnW", [1, 1, attn_size, attention_vec_size])
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = variable_scope.get_variable("AttnV", [attention_vec_size])

            y = linear(query, attention_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum( v * math_ops.tanh(hidden_features + y), [2, 3]) 
            a = nn_ops.softmax(s + attn_mask)
            # Now calculate the attention-weighted vector d.
            d = math_ops.reduce_sum( array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            d = array_ops.reshape(d, [-1, attn_size])  #remember this size
            return d, a

    # Write history memory
    def __write_memory(self, his_mem, enc_states, global_trace, step):
        with variable_scope.variable_scope("write_memory"):
            mem_slots = his_mem.get_shape()[1].value
            mem_size = his_mem.get_shape()[2].value

            for i, state in enumerate(enc_states):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                # Concatenate history memory with the null slot
                tmp_mem = array_ops.concat([his_mem, tf.identity(self.null_mem)], axis=1)

                hidden = array_ops.reshape(tmp_mem, [-1, mem_slots+1, 1, mem_size]) 
                k = variable_scope.get_variable("AttnW", [1, 1, mem_size, mem_size])
                mem_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                v = variable_scope.get_variable("AttnV", [mem_size])

                mstate = state
                y = linear([flatten_query(mstate), global_trace], mem_size, True, scope = "query_trans")
                y = array_ops.reshape(y, [-1, 1, 1, mem_size])
                s = math_ops.reduce_sum(v * math_ops.tanh(mem_features + y), [2, 3]) 
                random_mask = 1.0 - tf.sign(math_ops.reduce_sum(tf.abs(tmp_mem), axis=2))

                # The random_mask shows if a slot is empty, 1 empty, 0 not empty.
                # The null mask is 1 if there is at least 1 empty slot.
                null_mask = random_mask[:, 0:self.hps.his_mem_slots]
                null_mask = math_ops.reduce_sum(null_mask, axis=1)
                null_mask = tf.sign(null_mask)

                bias = self.random_bias * random_mask
                max_bias = tf.reduce_max(bias, axis=1)
                max_bias = tf.expand_dims(max_bias, axis=1)
                bias = tf.divide(bias, max_bias + 1e-12)
                
                max_s = tf.expand_dims(math_ops.reduce_max(s, axis=1), axis=1)

                thred1 = tf.ones([self.b_size, self.hps.his_mem_slots+1], dtype=tf.float32)
                thred2 = tf.zeros([self.b_size, self.hps.his_mem_slots+1], dtype=tf.float32)
                thred = tf.where(tf.equal(null_mask, 1), thred1, thred2)

                bias1 = bias * tf.abs(max_s) * thred
                s1 = s + bias1
                a = nn_ops.softmax(s1)

                max_val = tf.reduce_max(a, axis=1)
                max_val = tf.expand_dims(max_val, axis=1)

                if self.mode == 'train':
                    float_mask0 = tf.tanh(self.gama * (a - max_val)) + 1.0
                elif self.mode == 'decode':
                    float_mask0 = tf.sign(a - max_val) + 1.0

                mask = self.write_masks[step][i]
                float_mask = tf.multiply(mask, float_mask0)
                float_mask = tf.expand_dims(float_mask, axis=2)
                #print (np.shape(float_mask))

                w_states = tf.tile(mstate, [1, mem_slots])
                w_states = array_ops.reshape(w_states, [-1, mem_slots, mem_size])
                
                final_mask = float_mask[:, 0:self.hps.his_mem_slots, :]
                #print (final_mask.get_shape())
                his_mem = (1.0 - final_mask) * his_mem + final_mask * w_states

            return his_mem

    def __topic_trace_update(self, topic_trace, key_align, key_states):
        with variable_scope.variable_scope("topic_trace_update"):
            key_used = math_ops.reduce_mean(tf.multiply(key_states, 
                tf.expand_dims(key_align, axis=2)), axis=1)
            new_topic_trace = linear([topic_trace[:, 0:self.hps.topic_trace_size], key_used], 
                self.hps.topic_trace_size, True)
            new_topic_trace = tf.tanh(new_topic_trace)
            attn_los = topic_trace[:, self.hps.topic_trace_size:] + key_align

            fin_topic_trace = array_ops.concat([new_topic_trace, attn_los], 1)
            return fin_topic_trace

    def __global_trace_update(self, global_trace, enc_states):
        with variable_scope.variable_scope("global_trace_update", reuse=None):
            state = math_ops.reduce_mean(enc_states, axis=1)            
            new_global_trace = math_ops.tanh(linear([global_trace, state], 
                self.hps.global_trace_size, True))
            new_global_trace = array_ops.reshape(new_global_trace, [-1, self.hps.global_trace_size])
            return new_global_trace

    #_____________________________________________
    # Functions for training or generation

    def step(self, session, data_dic, forward_only):
        '''For training one batch'''
        keep_prob = 1.0 if forward_only else 0.75

        input_feed = {}
        input_feed[self.keep_prob] = keep_prob
        
        for step in xrange(self.hps.key_slots):
            # NOTE: each topic word must consist of no more than 2 characters
            for j in xrange(2):
                input_feed[self.key_inps[step][j].name] = data_dic['key_inps'][step][j]
        input_feed[self.key_mask.name] = data_dic['key_mask']

        for step in xrange(0, self.hps.sens_num):
            if len(data_dic['enc_inps'][step]) != self.enc_len:
                raise ValueError("length must be equal %d != %d." % 
                    (len(data_dic['enc_inps'][step]), self.enc_len))

            if len(data_dic['dec_inps'][step]) != self.dec_len:
                raise ValueError("length must be equal %d != %d. " %
                 (len(data_dic['dec_inps'][step]), self.dec_len))
            
            if len(data_dic['trg_weights'][step]) != self.dec_len:
                raise ValueError(" length must be equal %d != %d." %
                    (len(data_dic['trg_weights'][step]), self.dec_len))

        
            for l in xrange(self.enc_len):
                input_feed[self.enc_inps[step][l].name] = data_dic['enc_inps'][step][l]
                input_feed[self.write_masks[step][l].name] = data_dic['write_masks'][step][l]
            for l in xrange(self.dec_len):
                input_feed[self.dec_inps[step][l].name] = data_dic['dec_inps'][step][l]
                input_feed[self.trg_weights[step][l].name] = data_dic['trg_weights'][step][l]
                input_feed[self.len_inps[step][l].name] = data_dic['len_inps'][step][l]
                input_feed[self.ph_inps[step][l].name] = data_dic['ph_inps'][step][l]

            last_target = self.dec_inps[step][self.dec_len].name
            input_feed[last_target] = np.ones([self.hps.batch_size], dtype=np.int32) * self.hps.PAD_ID
            input_feed[self.enc_mask[step].name] =data_dic['enc_mask'][step]

        output_feed = []       
        for step in xrange(0, self.hps.sens_num):
            for l in xrange(self.dec_len):  # Output logits.
                output_feed.append(self.outputs[step][l])

        if not forward_only:
            output_feed += [self.update, self.gen_loss] 
        else:
            output_feed += [self.gen_loss]  # Loss for this batch.

        outputs = session.run(output_feed, input_feed)

        logits = []
        for step in xrange(0, self.hps.sens_num):
            logits.append(outputs[step*self.dec_len:(step+1)*self.dec_len])

        n = self.dec_len * self.hps.sens_num
        if not forward_only:
            return logits, outputs[n+1]
        else:
            return logits, outputs[n]

    #----------------------------------------
    # Some apis for beam search
    def key_memory_computer(self, session, key_inps, key_mask):
        input_feed = {}
        input_feed[self.keep_prob] = 1.0
        for step in xrange(self.hps.key_slots):
            for j in xrange(2):
                input_feed[self.key_inps[step][j].name] = key_inps[step][j]
        input_feed[self.key_mask.name] = key_mask

        output_feed = [self.key_initial_state, self.key_states]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # key_initial_state, key_states

    def encoder_computer(self, session, enc_inps, enc_mask):
        assert self.enc_len == len(enc_inps) 
        input_feed = {}
        input_feed[self.keep_prob] = 1.0
        for l in xrange(self.enc_len):
            input_feed[self.enc_inps[0][l].name] = enc_inps[l]

        input_feed[self.enc_mask[0].name] = enc_mask
        output_feed = [self.enc_state, self.attn_states]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # encoder_state, attention_states
        

    def decoder_state_computer(self, sess, dec_inp, len_inp, ph_inp, prev_state,
        attn_states, key_states, his_mem, global_trace, enc_mask, key_mask,
         his_mem_mask, topic_trace):
        input_feed = {}
        input_feed[self.keep_prob] = 1.0

        input_feed[self.dec_inps[0][0].name] = dec_inp
        input_feed[self.len_inps[0][0].name] = len_inp
        input_feed[self.ph_inps[0][0].name] = ph_inp

        input_feed[self.beam_attn_states.name] = attn_states
        input_feed[self.enc_mask[0].name] = enc_mask
        input_feed[self.beam_initial_state.name] = prev_state
        input_feed[self.beam_key_states.name] = key_states
        input_feed[self.key_mask.name] = key_mask
        input_feed[self.beam_global_trace.name] = global_trace
        input_feed[self.beam_topic_trace.name] = topic_trace

        input_feed[self.beam_his_mem.name] = his_mem
        input_feed[self.beam_his_mem_mask.name] = his_mem_mask

        output_feed = [self.next_out[0], self.next_state[0], self.next_align[0][0]]

        outputs = sess.run(output_feed, input_feed)

        return outputs[0], outputs[1], outputs[2]# next_output, next_state, next_align

    def topic_trace_computer(self, session, key_states, prev_topic_trace, key_align):
        input_feed = {}
        input_feed[self.keep_prob] = 1.0
        input_feed[self.beam_key_states.name] = key_states
        input_feed[self.beam_topic_trace.name] = prev_topic_trace
        input_feed[self.beam_key_align.name] = key_align
        output_feed = [self.new_topic_trace]

        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def global_trace_computer(self, session, prev_global_trace, prev_attn_states):
        input_feed = {}
        input_feed[self.keep_prob] = 1.0
        input_feed [self.beam_global_trace] = prev_global_trace
        input_feed[self.beam_attn_states] = prev_attn_states
        output_feed = [self.new_global_trace]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def his_mem_computer(self, session, his_mem, prev_enc_states, mem_write_mask, global_trace):
        input_feed = {}
        input_feed[self.keep_prob] = 1.0
        input_feed[self.beam_his_mem.name] = his_mem

        for l in xrange(self.enc_len):
            input_feed[self.beam_mem_states[l]] = prev_enc_states[l]
            input_feed[self.write_masks[0][l]] = mem_write_mask[l]
        input_feed [self.beam_global_trace.name] = global_trace

        output_feed = [self.new_his_mem] 
        outputs = session.run(output_feed, input_feed)

        return outputs[0]