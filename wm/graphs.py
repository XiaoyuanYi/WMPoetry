from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import layers as layers_lib

class WorkingMemoryModel(object):
    def __init__(self, hps, f_idxes=None, init_emb=None):
        # Create the model
        self.hps = hps
        self.enc_len = hps.bucket[0]
        self.dec_len = hps.bucket[1]
        self.device = self.hps.device
        self.global_step = tf.Variable(0, trainable=False)

        self.f_idxes = f_idxes

        self.learning_rate = tf.Variable(float(self.hps.learning_rate), trainable=False)
        self.learning_rate_decay_op = \
            self.learning_rate.assign(self.learning_rate * self.hps.decay_rate)

        self.__build_placeholders()

        # Build modules and layers
        with tf.device(self.device):
            self.layers = {}

            self.layers['word_emb'] = layers_lib.Embedding(
                self.hps.vocab_size, self.hps.emb_size,name="word_emb")

            # Build genre embedding
            # NOTE: We set fixed 36 phonology categories
            self.layers['ph_emb'] = layers_lib.Embedding(
                36, self.hps.ph_emb_size, name="ph_emb")

            self.layers['len_emb'] = layers_lib.Embedding(
                self.dec_len+1, self.hps.len_emb_size, name="len_emb")

            # Build Encoder
            self.layers['enc'] = layers_lib.BidirEncoder(
                self.hps.hidden_size, self.keep_prob, name="enc")

            # The decoder cell
            self.layers['dec'] = layers_lib.Decoder(
                self.hps.hidden_size, self.hps.vocab_size, self.keep_prob, name="dec")

            # History memory reading and writing layers
            self.layers['attn_read'] = layers_lib.AttentionLayer("attn_read")
            self.layers['attn_write'] = layers_lib.AttnWriteLayer(self.hps.his_mem_slots, self.hps.mem_size, "attn_write")

            # global trace update
            self.layers['global_trace'] = layers_lib.GTraceLayer(3, self.hps.global_trace_size, "global_trace")

            # NOTE: a layer to compress the states to a smaller size for larger number of slots
            self.layers['mlp_compress'] = layers_lib.MLPLayer([self.hps.mem_size], 
                ['tanh'], keep_prob=self.keep_prob, name='mlp_compress')

            self.layers['mlp_key_initial'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                ['tanh'], keep_prob=self.keep_prob, name='mlp_key_initial')
            self.layers['mlp_enc_initial'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                ['tanh'], keep_prob=self.keep_prob, name='mlp_enc_initial')
            self.layers['mlp_dec_merge'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                [None], keep_prob=self.keep_prob, name='mlp_dec_merge')
            self.layers['mlp_topic_trace'] = layers_lib.MLPLayer([self.hps.topic_trace_size], 
                ['tanh'], keep_prob=self.keep_prob, name='mlp_topic_trace')
            self.layers['mlp_init_null'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                [None], keep_prob=self.keep_prob, name='mlp_init_null')

            # loss
            self.layers['softmax_loss'] = layers_lib.SoftmaxLoss(name='softmax_loss')

            # for pre-training
            self.layers['mlp_dec_merge_ae'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                [None], keep_prob=self.keep_prob, name='mlp_dec_merge_ae')    


    def __build_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.gama = tf.placeholder(tf.float32,  shape=[1])

        sens_num = self.hps.sens_num
        self.enc_inps = [[] for x in range(sens_num)]
        self.dec_inps = [[] for x in range(sens_num)]
        self.enc_mask  = [[] for x in range(sens_num)]
        self.ph_inps = [[] for x in range(sens_num)]
        self.len_inps = [[] for x in range(sens_num)]
        self.write_masks = [[] for x in range(sens_num)]

        for step in range(0, sens_num):
            for i in range(self.enc_len):
                self.enc_inps[step].append(tf.placeholder(tf.int32, shape=[None], name="enc{0}_{1}".format(step, i)))
                self.write_masks[step].append(tf.placeholder(tf.float32, shape=[None, 1], name="write_masks{0}_{1}".format(step, i)))

            # The dec_len is added 1 since we build the targets by shifting one position of the decoder inputs
            for i in range(self.dec_len + 1):
                self.dec_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="dec{0}_{1}".format(step, i)))
                self.len_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="len_inps{0}_{1}".format(step, i)))
                self.ph_inps[step].append(tf.placeholder(tf.int32, shape=[None],name="ph_inps{0}_{1}".format(step, i)))

            self.enc_mask[step] = tf.placeholder(tf.float32, shape=[None, self.enc_len, 1], name="enc_mask{0}".format(step))

        self.key_inps = [[] for x in range(self.hps.key_slots)]
        for i in range(self.hps.key_slots):
            # NOTE: We set that each keyword must consist of no more than 2 characters
            for j in range(2):
                self.key_inps[i].append(tf.placeholder(tf.int32, shape=[None], name="key{0}_{1}".format(i, j)))
        self.key_mask = tf.placeholder(tf.float32, shape=[None, self.hps.key_slots, 1], name="key_mask")

        self.trg_weights = [[] for x in range(sens_num)]
        for step in range(0, sens_num):
            for i in range(self.dec_len + 1):
                self.trg_weights[step].append(tf.placeholder(tf.float32, shape=[None], name="weight{0}_{1}".format(step, i)))

        self.targets = [[] for x in range(sens_num)]
        for step in range(0, sens_num):
            self.targets[step] = [self.dec_inps[step][i + 1] for i in range(len(self.dec_inps[step]) - 1)]
        
        # -----------------------------------------------
        # For beam search
        self.beam_key_align = tf.placeholder(tf.float32, 
            shape=[None, self.hps.key_slots], name="beam_key_align")
        self.beam_attn_states = tf.placeholder(tf.float32, 
            shape=[None, self.enc_len, self.hps.mem_size], name="beam_attn_states")
        self.beam_initial_state = tf.placeholder(tf.float32, 
            shape=[None, self.hps.hidden_size] , name="beam_initial_state")
        self.beam_key_states = tf.placeholder(tf.float32, 
            shape=[None, self.hps.key_slots, self.hps.mem_size], name="beam_key_states")
        self.beam_global_trace = tf.placeholder(tf.float32, 
            shape=[None, self.hps.global_trace_size], name="beam_global_trace")
        
        self.beam_his_mem = tf.placeholder(tf.float32, 
            [None, self.hps.his_mem_slots, self.hps.mem_size], name="beam_his_mem")
        self.beam_his_mem_mask = tf.placeholder(tf.float32, 
            [None, self.hps.his_mem_slots], name="beam_his_mem_mask")
        self.beam_topic_trace = tf.placeholder(tf.float32, 
            [None, self.hps.topic_trace_size + self.hps.key_slots], name="beam_topic_trace")

        # For history memory write
        self.beam_enc_outs = []
        for i in range(self.enc_len):
            self.beam_enc_outs.append(tf.placeholder(tf.float32, 
                shape=[None, self.hps.mem_size],name="beam_enc_outs{0}".format(i)))

    def training(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):

            normed_outs_vec, loss_vec = self.__build_graph()
            gen_loss = math_ops.reduce_mean(loss_vec) 
            train_op, regular_loss = self.__optimization(gen_loss, self.hps.l2_weight)
            
        return train_op, normed_outs_vec, gen_loss, regular_loss, self.global_step

    def pre_training(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):
            normed_outs_vec, gen_loss  = self.__build_pretrain_graph()
            train_op, regular_loss = self.__optimization(gen_loss, self.hps.l2_weight)
            
        return train_op, normed_outs_vec, gen_loss, regular_loss, self.global_step

    @property
    def pretrained_variables(self):
        variables = self.layers['word_emb'].trainable_weights + self.layers['ph_emb'].trainable_weights \
            + self.layers['len_emb'].trainable_weights + self.layers['enc'].trainable_weights \
            + self.layers['dec'].trainable_weights 

        return variables

    def __build_pretrain_graph(self):
        '''
        We pre-train the encoder, deocder and embeddings of the model
            by training a simple sequence-to-sequence model
        '''
        print (self.device)
        with tf.device(self.device):
            emb_enc_inps = [self.layers['word_emb'](x) for x in  self.enc_inps[0]]
            emb_dec_inps = [self.layers['word_emb'](x) for x in  self.dec_inps[0]]

            emb_ph_inps = [self.layers['ph_emb'](x) for x  in self.ph_inps[0]]
            emb_len_inps = [self.layers['len_emb'](x)  for x in self.len_inps[0]]

            emb_genre = []
            for i in range(self.dec_len+1):
                emb_genre.append(array_ops.concat([emb_ph_inps[i], emb_len_inps[i]], 1) )

            # Encoder
            _, _, enc_state_bw = self.layers['enc'](emb_enc_inps)
            init_state = self.layers['mlp_enc_initial'](enc_state_bw)

            dec_outs, normed_dec_outs  = [], []

            state = init_state
            for i, inp in enumerate(emb_dec_inps[ : self.dec_len]):
                # Run the GRU
                x = self.layers['mlp_dec_merge_ae']([inp, emb_genre[i]])
                out, state = self.layers['dec'](state, x)

                dec_outs.append(tf.identity(out))
                normed_dec_outs.append(tf.identity(tf.nn.softmax(out)))

            loss = self.layers['softmax_loss'](dec_outs, self.targets[0][: self.dec_len], self.trg_weights[0][ : self.dec_len])

        return normed_dec_outs, loss


    def __build_graph(self):
        print (self.device)
        with tf.device(self.device):
            self.emb_enc_inps = [ [self.layers['word_emb'](x) for x in enc_inp] for enc_inp in self.enc_inps]
            self.emb_dec_inps = [ [self.layers['word_emb'](x) for x in dec_inp] for dec_inp in self.dec_inps]
            emb_key_inps = [ [self.layers['word_emb'](x) for x in self.key_inps[i] ] for i in range(0, self.hps.key_slots)]
            emb_ph_inps = [ [self.layers['ph_emb'](x) for x in ph_inp] for ph_inp in self.ph_inps]
            emb_len_inps = [ [self.layers['len_emb'](x)  for x in len_inp] for len_inp in self.len_inps]

            # Build null memory slot. We initialize the slot with an average of stop words
            #    by supposing that the model could learn to ignore these  words
            null_idxes = tf.constant(self.f_idxes)
            null_emb = self.layers['word_emb'](null_idxes)
            null_mem = self.layers['mlp_init_null'](math_ops.reduce_mean(null_emb, axis=0))
            null_mem = tf.expand_dims(null_mem, axis=0)
            null_mem = tf.expand_dims(null_mem, axis=1)
            self.null_mem = tf.tile(null_mem, [tf.shape(self.enc_inps[0][0])[0], 1, 1])

            # Concatenate phonology embedding and length embedding to 
            #   form the genre embedding
            self.emb_genre = [[] for x in range(self.hps.sens_num)]
            for step in range(0, self.hps.sens_num):
                for i in range(self.dec_len+1):
                    self.emb_genre[step].append(array_ops.concat([emb_ph_inps[step][i], emb_len_inps[step][i]], 1) )

            # Build the topic memory
            key_initial_state, key_states = self.__build_key_memory(emb_key_inps)

            global_trace = array_ops.zeros([self.hps.batch_size, 
                self.hps.global_trace_size], dtype=tf.float32)
            
            # history memory
            his_mem = array_ops.zeros([self.hps.batch_size, self.hps.his_mem_slots, 
                self.hps.mem_size], dtype=tf.float32)
            
            topic_trace = array_ops.zeros([self.hps.batch_size, 
                self.hps.topic_trace_size+self.hps.key_slots], dtype=tf.float32)

            his_mem_mask = array_ops.zeros([self.hps.batch_size, self.hps.his_mem_slots], dtype=tf.float32)
            
            normed_outs_vec, loss_vec = [], []
            
            for step in range(0, self.hps.sens_num):
                if step > 0:
                    key_initial_state = None
                normed_outs, loss, global_trace, his_mem, topic_trace, \
                    = self.__build_seq2seq(global_trace, key_initial_state, 
                    key_states, his_mem, his_mem_mask, topic_trace, step)

                if step >=1:
                    his_mem_sum = math_ops.reduce_sum(tf.abs(his_mem), axis=2)
                    his_mem_bool = tf.not_equal(his_mem_sum, 0)
                    his_mem_mask = tf.cast(his_mem_bool, tf.float32)

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

        enc_mask_squ = tf.squeeze(self.enc_mask[step], axis = 2)
        key_mask_squ= tf.squeeze(self.key_mask, axis = 2)
        # Note this order: 1: history memory, 2: key memory, 3: local memory
        total_mask = array_ops.concat( [his_mem_mask, key_mask_squ, enc_mask_squ], axis=1)
        total_attn_states = array_ops.concat( [his_mem, key_states, attn_states], axis=1)

        normed_dec_outs, dec_outs, dec_states, attn_weights = self.__build_decoder(
            self.emb_dec_inps[step][ : self.dec_len], total_attn_states, total_mask,
            global_trace, initial_state, self.emb_genre[step], topic_trace)

        # Update topic trace 
        concat_aligns = array_ops.concat(attn_weights, 1)
        # concat_aligns: [batch_size, dec_len, memory_slots]
        key_align = concat_aligns[:, :, self.hps.his_mem_slots:self.hps.his_mem_slots+self.hps.key_slots]
        key_align = math_ops.reduce_mean(key_align, axis=1)
        new_topic_trace = self.__topic_trace_update(topic_trace, key_align, key_states)

        # Write the history memory
        new_his_mem = self.layers['attn_write'](his_mem, enc_outs, global_trace, 
            self.write_masks[step], self.null_mem, self.gama)

        # Update global trace
        new_global_trace = self.layers['global_trace'](global_trace, attn_states)
        loss = self.layers['softmax_loss'](dec_outs, self.targets[step][: self.dec_len], self.trg_weights[step][ : self.dec_len])
        return normed_dec_outs, loss, new_global_trace, new_his_mem, new_topic_trace

    def __build_key_memory(self, emb_key_inps):
        key_states = []
        for i in range(0, self.hps.key_slots):
            _, state_fw, state_bw = self.layers['enc'](emb_key_inps[i]) 
            key_state = array_ops.concat([state_fw, state_bw], 1)
            key_state = self.layers['mlp_compress'](key_state)
            key_states.append(key_state)

        key_states = [array_ops.reshape(e, [-1, 1, self.hps.mem_size]) for e in key_states]
        key_states = array_ops.concat(key_states, 1)
        # The pads are masked
        key_states = tf.multiply(self.key_mask, key_states)

        all_state = math_ops.reduce_sum(key_states, axis=1)
        init_state = self.layers['mlp_key_initial'](all_state)

        return init_state, key_states
            
    def __build_encoder(self, step):
        x = self.emb_enc_inps[step][:self.enc_len]
        enc_outs, enc_state_fw, enc_state_bw = self.layers['enc'](x)
        init_state = self.layers['mlp_enc_initial'](enc_state_bw)

        compress_states = [self.layers['mlp_compress'](e) for e in enc_outs]
        top_states = [array_ops.reshape(e, [-1, 1, self.hps.mem_size]) for e in compress_states]
        attention_states = array_ops.concat(top_states, 1)
        masked_attn_states = tf.multiply(self.enc_mask[step], attention_states)

        return init_state, masked_attn_states, compress_states

    def __build_decoder(self, dec_inps, total_attn_states, total_mask, global_trace,
        initial_state, genre, topic_trace):
        if not total_attn_states.get_shape()[1:3].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attn_states must be known: %s" % total_attn_states.get_shape())

        dec_outs, dec_states, attn_weights  = [], [], []
        normed_dec_outs = []

        # build attn_mask
        imask = 1.0 - total_mask
        attn_mask = tf.where(tf.equal(imask, 1), tf.ones_like(imask) * (-float('inf')), imask)

        state = initial_state
        for i, inp in enumerate(dec_inps):
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            # Read memory
            query = array_ops.concat([state, global_trace, topic_trace], axis=1)
            attns, align = self.layers['attn_read'](total_attn_states, query, attn_mask)

            x = self.layers['mlp_dec_merge']([inp, attns, genre[i], global_trace])

            # Run the GRU
            out, state = self.layers['dec'](state, x)

            dec_outs.append(tf.identity(out))
            normed_dec_outs.append(tf.identity(tf.nn.softmax(out)))
            align = tf.expand_dims(align, axis=1)
            attn_weights.append(align)
            dec_states.append(tf.identity(state))

        return  normed_dec_outs, dec_outs, dec_states, attn_weights

    def __topic_trace_update(self, topic_trace, key_align, key_states):
        # topic_trace: [batch_size, topic_trace_size+key_slots]
        # key_align: [batch_size, key_slots]
        # key_states: [batch_size, key_slots, mem_size]
        key_used = math_ops.reduce_mean(tf.multiply(key_states, 
            tf.expand_dims(key_align, axis=2)), axis=1)

        new_topic_trace = self.layers['mlp_topic_trace'](
            [topic_trace[:, 0:self.hps.topic_trace_size], key_used])
        attn_los = topic_trace[:, self.hps.topic_trace_size:] + key_align

        fin_topic_trace = array_ops.concat([new_topic_trace, attn_los], 1)
        return fin_topic_trace

    def __optimization(self, gen_loss, l2_weight):
        params = tf.trainable_variables()                
        regularizers = []

        for param in params:
            regularizers.append(tf.nn.l2_loss(param))
                
        regular_loss = math_ops.reduce_sum(regularizers)
        
        total_loss = gen_loss + l2_weight * regular_loss
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(total_loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.hps.max_gradient_norm)
                
        train_op = opt.apply_gradients( zip(clipped_gradients, params), global_step=self.global_step)
        return train_op, regular_loss

    #-----------------------------------
    def build_eval_graph(self):
        '''
        Build the evaluation graph for testing and generation
        '''
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):
            # key computer
            # key_inps and key_mask are required
            emb_key_inps = [ [self.layers['word_emb'](x) for x in self.key_inps[i] ] for i in range(0, self.hps.key_slots)]
            self.key_initial_state, self.key_states = self.__build_key_memory(emb_key_inps)

            # encoder 
            # enc_inps[0] and enc_mask[0] are required
            self.emb_enc_inps = [[self.layers['word_emb'](x) for x in self.enc_inps[0]]]
            self.enc_state, self.attn_states, _ = self.__build_encoder(0)
            
            # decoder
            enc_mask_squ =  tf.squeeze(self.enc_mask[0], axis=2)
            key_mask_squ =  tf.squeeze(self.key_mask, axis=2)
            # Note this order: 1: history memory, 2: key memory, 3: local memory
            total_mask = array_ops.concat([self.beam_his_mem_mask, key_mask_squ, enc_mask_squ], 1)
            total_attn_states = array_ops.concat([self.beam_his_mem, self.beam_key_states, self.beam_attn_states], 1)

            # build genre
            emb_ph_inp = self.layers['ph_emb'](self.ph_inps[0][0])
            emb_len_inp = self.layers['len_emb'](self.len_inps[0][0])

            emb_genre = [array_ops.concat([emb_ph_inp, emb_len_inp], 1)]
            emb_dec_inp = [self.layers['word_emb'](self.dec_inps[0][0])]

            normed_dec_outs, _, dec_states, attn_weights = self.__build_decoder(emb_dec_inp,
                total_attn_states, total_mask, self.beam_global_trace, self.beam_initial_state, emb_genre, self.beam_topic_trace)
            self.next_out = normed_dec_outs[0]
            self.next_state = dec_states[0]
            self.next_align = attn_weights[0]
            
            # memory write
            null_idxes = tf.constant(self.f_idxes)
            null_emb = self.layers['word_emb'](null_idxes)
            null_mem = self.layers['mlp_init_null'](math_ops.reduce_mean(null_emb, axis=0))
            null_mem = tf.expand_dims(null_mem, axis=0)
            null_mem = tf.expand_dims(null_mem, axis=1)
            null_mem = tf.tile(null_mem, [tf.shape(self.beam_his_mem)[0], 1, 1])

            self.new_his_mem = self.layers['attn_write'](self.beam_his_mem, self.beam_enc_outs, self.beam_global_trace, 
                self.write_masks[0], null_mem, self.gama)

            self.new_topic_trace = self.__topic_trace_update(self.beam_topic_trace, self.beam_key_align, self.beam_key_states)
            self.new_global_trace = self.layers['global_trace'](self.beam_global_trace, self.beam_attn_states)