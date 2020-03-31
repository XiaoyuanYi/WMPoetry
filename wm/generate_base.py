from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import graphs
from tool import PoetryTool
from DataTool import data_tool
from config import hps
import numpy as np

class Generator(object):

    def __init__(self, beam_size, model_path=None):
        # Construct hyper-parameter
        self.hps = hps
        self.dtool = data_tool
        self.beam_size = beam_size
        if hps.init_emb == '':
            self.init_emb = None
        else:
            self.init_emb = np.load(self.hps.init_emb)
            print ("init_emb_size: %s" % str(np.shape(self.init_emb)))

        self.tool = PoetryTool(sens_num=hps.sens_num,
            key_slots=hps.key_slots, enc_len=hps.bucket[0],
            dec_len=hps.bucket[1])
        self.tool.load_dic(hps.vocab_path, hps.ivocab_path)

        vocab_size = self.tool.get_vocab_size()
        assert vocab_size > 0
        self.hps = self.hps._replace(vocab_size=vocab_size, 
            batch_size=beam_size)

        f_idxes = self.tool.build_fvec()
        self.model = graphs.WorkingMemoryModel(self.hps, f_idxes)
        self.model.build_eval_graph()

        self.PAD_ID, self.UNK_ID, self.B_ID, self.E_ID, _ \
            = self.tool.get_special_IDs()

        self.enc_len = self.hps.bucket[0]
        self.dec_len = self.hps.bucket[1]
        self.topic_trace_size = self.hps.topic_trace_size
        self.key_slots = self.hps.key_slots
        self.his_mem_slots = self.hps.his_mem_slots
        self.mem_size = self.hps.mem_size
        self.global_trace_size = self.hps.global_trace_size
        self.hidden_size = self.hps.hidden_size

        self.sess = tf.InteractiveSession()

        self.load_model(model_path)
        self.__buildPH()

    def load_model(self, model_path):
        """load parameters in session."""
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)

        if model_path is None:
            ckpt = tf.train.get_checkpoint_state(self.hps.model_path)
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % 
                    ckpt.model_checkpoint_path)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError("%s not found! " % ckpt.model_checkpoint_path)
        else:
            print("Reading model parameters from %s" % model_path)
            saver.restore(self.sess, model_path)

    def __buildPH(self):
        self.__PHDic = self.dtool.buildPHDicForIdx(
            copy.deepcopy(self.tool.get_vocab()))
        
    def addtionFilter(self, trans, pos):
        pos -= 1
        preidx = range(0, pos)
        batch_size = len(trans)
        forbidden_list = [[] for _ in range(0, batch_size)]

        for i in range(0, batch_size):
            prechar = [trans[i][c] for c in preidx]
            forbidden_list[i] = prechar

        return forbidden_list

    def beam_select(self, probs, trans, k, trg_len, beam_size, repeatidxvec, ph):
        V = np.shape(probs)[1]  # vocabulary size
        n_samples = np.shape(probs)[0]
        if k == 1:
            n_samples = beam_size

        # trans_indices, word_indices, costs
        hypothesis = []  # (char_idx, which beam, prob)
        cost_eps = float(1e5)

        # Control inner repeat
        forbidden_list = self.addtionFilter(trans, k)
        for i in range(0, np.shape(probs)[0]):
            probs[i, forbidden_list[i]] = cost_eps

        # Control global repeat
        probs[:, repeatidxvec] = cost_eps

        # hard control for genre
        if ph != 0:
            #print (k, gl)
            probs *= cost_eps
            probs[:, self.__PHDic[ph]] /= float(cost_eps)

        flat_next_costs = probs.flatten()
        best_costs_indices = np.argpartition(
            flat_next_costs.flatten(), n_samples)[:n_samples]

        trans_indices = [int(idx)
                         for idx in best_costs_indices / V]  # which beam line
        word_indices = best_costs_indices % V
        costs = flat_next_costs[best_costs_indices]

        for i in range(0, n_samples):
            hypothesis.append((word_indices[i], trans_indices[i], costs[i]))

        return hypothesis

    def beam_search(self, sen, len_inps, ori_key_states, key_initial_state, ori_topic_trace, 
        ori_his_mem, ori_his_mem_mask, ori_global_trace, enc_mask, ori_key_mask, repeatidxvec, phs):
        trg_length = len(phs)
        beam_size = self.beam_size

        enc_state, ori_attn_states = self.encoder_computer(sen, enc_mask)
        enc_mask = np.tile(enc_mask, [beam_size, 1, 1])
        key_mask = np.tile(ori_key_mask, [beam_size, 1, 1])

        attn_states = copy.deepcopy(ori_attn_states)
        key_states = copy.deepcopy(ori_key_states)
        topic_trace= copy.deepcopy(ori_topic_trace)
        global_trace = copy.deepcopy(ori_global_trace)
        his_mem = copy.deepcopy(ori_his_mem)
        his_mem_mask = copy.deepcopy(ori_his_mem_mask)

        fin_trans, fin_costs, fin_align = [], [], []

        trans = [[] for i in range(0, beam_size)]
        costs = [0.0]

        key_align = []
        for i in range(beam_size):
            key_align.append(np.zeros([1, self.key_slots], dtype=np.float32))

        state = enc_state
        if not (key_initial_state is None):
            state = key_initial_state
        inp = np.array([self.B_ID]*beam_size)

        ph_inp = [phs[0]]*beam_size
        specify_len = len_inps[0]

        output, state, alignments = self.decoder_computer(inp, specify_len, ph_inp, state, 
            attn_states, key_states, his_mem, global_trace, enc_mask, key_mask, 
            his_mem_mask, topic_trace)

        n_samples = beam_size

        for k in range(1, trg_length+4):
            if n_samples == 0:
                break

            if k == 1:
                output = output[0, :]
          
            log_probs = np.log(output)
            next_costs = np.array(costs)[:, None] - log_probs

            # Form a beam for the next iteration
            new_trans = [[] for i in range(0, n_samples)]
            new_costs = np.zeros(n_samples, dtype="float32")
            new_states = np.zeros((n_samples, self.hidden_size), dtype="float32")
            new_align = [[] for i in range(0, n_samples)]

            inputs = np.zeros(n_samples, dtype=np.int32)
            
            ph_require = phs[k-1] if k <= len(phs) else 0
            
            #print (gl_require)
            hypothesis = self.beam_select(
                next_costs, trans, k, trg_length, n_samples, repeatidxvec, ph_require)

            for i, (next_word, orig_idx, next_cost) in enumerate(hypothesis):
                #print("%d %d %d %f %s" % (i, next_word, orig_idx, next_cost))
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                align_start = self.his_mem_slots
                align_end = self.his_mem_slots+self.key_slots
                current_align = alignments[orig_idx, :, align_start:align_end]
                new_align[i] = np.concatenate((key_align[orig_idx], current_align), axis=0)
                new_states[i] = state[orig_idx, :]
                inputs[i] = next_word

            # Filter the sequences that end with end-of-sequence character
            trans, costs, indices, key_align = [], [], [], []

            for i in range(n_samples):
                if new_trans[i][-1] != self.E_ID:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                    key_align.append(new_align[i])
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    fin_align.append(new_align[i])

            if n_samples == 0:
                break

            inputs = inputs[indices]
            new_states = new_states[indices]
            attn_states = attn_states[indices, :, :]
            
            global_trace = global_trace[indices, :]
            enc_mask= enc_mask[indices, :, :]
            key_states = key_states[indices, :, :]
            his_mem = his_mem[indices, :, :]
            key_mask = key_mask[indices, :, :]
            his_mem_mask = his_mem_mask[indices, :]
            topic_trace = topic_trace[indices, :]

            if k >= np.shape(len_inps)[0]:
                specify_len = len_inps[np.shape(len_inps)[0]-1, indices]
            else:
                specify_len = len_inps[k, indices]

            if k >= len(phs):
                ph_inp = [0]*n_samples
            else:
                ph_inp = [phs[k]]*n_samples

            output, state, alignments = self.decoder_computer(inputs, specify_len,
                ph_inp, new_states, attn_states, key_states, his_mem, global_trace, enc_mask, key_mask,
                his_mem_mask,  topic_trace)
            

        for i in range(len(fin_align)):
            fin_align[i] = fin_align[i][1:, :]

        index = np.argsort(fin_costs)
        fin_align = np.array(fin_align)[index]
        fin_trans = np.array(fin_trans)[index]
        fin_costs = np.array(sorted(fin_costs))

        if len(fin_trans) == 0:
            index = np.argsort(costs)
            fin_align = np.array(key_align)[index]
            fin_trans = np.array(trans)[index]
            fin_costs = np.array(sorted(costs))

        return fin_trans, fin_costs, fin_align, ori_attn_states

    def get_new_global_trace(self, global_trace, ori_enc_states, beam_size):
        enc_states = np.expand_dims(ori_enc_states, axis=0)
        prev_global_trace = np.expand_dims(global_trace[0, :], 0)
        new_global_trace = self.global_trace_computer(prev_global_trace, enc_states)
        new_global_trace= np.tile(new_global_trace, [beam_size, 1])
        return new_global_trace

    def get_new_his_mem(self, ori_his_mem, attn_states, ori_global_trace, beam_size, src_len):
        #ori_his_mem [4, mem_size]
        his_mem = np.expand_dims(ori_his_mem, axis=0)
        enc_outs = []
        assert np.shape(attn_states)[0] == self.enc_len
        for i in range(0, self.enc_len):
            enc_outs.append(np.expand_dims(attn_states[i, :], 0))

        mask = [np.ones((1, 1))] * src_len + [np.zeros((1, 1))] * (self.enc_len-src_len)
        global_trace = np.expand_dims(ori_global_trace[0, :], 0)
        new_his_mem = self.his_mem_computer(his_mem, 
            enc_outs, mask, global_trace)

        new_his_mem = np.tile(new_his_mem, [beam_size, 1, 1])
        return new_his_mem

    def get_new_topic_trace(self, ori_topic_trace, key_align, ori_key_states, beam_size):
        key_states = np.expand_dims(ori_key_states[0, :, :], 0)
        topic_trace = np.expand_dims(ori_topic_trace, axis=0)
        key_align = np.mean(key_align, axis=0) 
        key_align = np.expand_dims(key_align, axis=0)
        new_topic_trace = self.topic_trace_computer(key_states, topic_trace, key_align)
        new_topic_trace = np.tile(new_topic_trace, [beam_size, 1])
        return new_topic_trace

    def generate_one(self, keystr, pattern):
        beam_size = self.beam_size
        ans, repeatidxes = [], []
        sens_num = len(pattern)

        keys = keystr.strip()
        print ("using keywords: %s" % (keystr))
        keys = keystr.split(" ")
        keys_idxes = [self.tool.chars2idxes(self.tool.line2chars(key)) for key in keys]

        # Here we set batch size=1 and then tile the results
        key_inps, key_mask = self.tool.gen_batch_key_beam(keys_idxes, batch_size=1)

        # Calculate initial_key state and key_states
        key_initial_state, key_states = self.key_memory_computer(key_inps, key_mask, beam_size)

        his_mem_mask = np.zeros([beam_size, self.his_mem_slots], dtype=np.float32)
        global_trace = np.zeros([beam_size, self.global_trace_size], dtype=np.float32)
        his_mem = np.zeros([beam_size, self.his_mem_slots, self.mem_size], dtype=np.float32)
        topic_trace = np.zeros([beam_size, self.topic_trace_size+self.key_slots], dtype=np.float32)
        
        # Generate the lines, line0 is an empty list
        sen = []
        for step in range(0, sens_num):
            print("generating %d line..." % (step+1))
            if step > 0:
                key_initial_state = None

            phs = pattern[step]
            trg_len = len(phs)
            src_len = len(sen)
            
            batch_sen, enc_mask, len_inps = self.tool.gen_enc_beam(sen, trg_len, batch_size=1)
            len_inps = np.tile(len_inps, [1, beam_size])

            trans, costs, align, attn_states= self.beam_search(batch_sen, len_inps, key_states,
                key_initial_state, topic_trace, his_mem, his_mem_mask, global_trace, enc_mask, key_mask,
                repeatidxes, phs)

            trans, costs, align, attn_states = self.pFilter(trans, costs, align, attn_states, trg_len)
            
            if len(trans) == 0:
                return [], ("line %d generation failed!" % (step+1))

            which = 0

            one_his_mem = his_mem[which, :, :]
            his_mem = self.get_new_his_mem(one_his_mem, 
                attn_states[which], global_trace, beam_size, src_len)

            if step >= 1:
                his_mem_mask = np.sum(np.abs(one_his_mem), axis=1)
                his_mem_mask = his_mem_mask != 0
                his_mem_mask = np.tile(his_mem_mask.astype(np.float32), [beam_size, 1])

            sentence = self.tool.beam_get_sentence(trans[which])
            sentence = sentence.strip()
            ans.append(sentence)
            attn_aligns = align[which][0:trg_len, :]
            topic_trace = self.get_new_topic_trace(topic_trace[which, :], attn_aligns,
                key_states, beam_size)
            global_trace = self.get_new_global_trace(global_trace, attn_states[which], beam_size)

            sentence = self.tool.line2chars(sentence)
            sen = self.tool.chars2idxes(sentence)
            repeatidxes = list(set(repeatidxes).union(set(sen)))

        return ans, "ok"

    def pFilter(self, trans, costs, align, states, trg_len):
        new_trans, new_costs, new_align, new_states = [], [], [], []

        for i in range(len(trans)):
            if len(trans[i]) < trg_len:
                continue

            tran = trans[i][0:trg_len]
            sen = self.tool.idxes2chars(tran)
            sen = "".join(sen)
            if trg_len > 4 and self.dtool.checkIfInLib(sen):
                continue
            new_trans.append(tran)
            new_align.append(align[i])
            new_states.append(states[i])
            new_costs.append(costs[i])

        return new_trans, new_costs, new_align, new_states

    # -----------------------------------------------------------------
    # Some apis for beam search
    def key_memory_computer(self, key_inps, key_mask, beam_size):
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        for step in range(self.key_slots):
            for j in range(2):
                input_feed[self.model.key_inps[step][j].name] = key_inps[step][j]
        input_feed[self.model.key_mask.name] = key_mask

        output_feed = [self.model.key_initial_state, self.model.key_states]
        [key_initial_state, key_states] = self.sess.run(output_feed, input_feed)
        key_initial_state = np.tile(key_initial_state, [beam_size, 1])
        key_states = np.tile(key_states, [beam_size, 1, 1])
        return key_initial_state, key_states

    def encoder_computer(self, enc_inps, enc_mask):
        assert self.enc_len == len(enc_inps) 
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        for l in range(self.enc_len):
            input_feed[self.model.enc_inps[0][l].name] = enc_inps[l]

        input_feed[self.model.enc_mask[0].name] = enc_mask
        output_feed = [self.model.enc_state, self.model.attn_states]
        [enc_state, attn_states] = self.sess.run(output_feed, input_feed)

        enc_state = np.tile(enc_state, [self.beam_size, 1])
        attn_states = np.tile(attn_states, [self.beam_size, 1, 1])

        return enc_state, attn_states

    def decoder_computer(self, dec_inp, len_inp, ph_inp, prev_state,
        attn_states, key_states, his_mem, global_trace, enc_mask, key_mask,
         his_mem_mask, topic_trace):
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0

        input_feed[self.model.dec_inps[0][0].name] = dec_inp
        input_feed[self.model.len_inps[0][0].name] = len_inp
        input_feed[self.model.ph_inps[0][0].name] = ph_inp

        input_feed[self.model.beam_attn_states.name] = attn_states
        input_feed[self.model.enc_mask[0].name] = enc_mask
        input_feed[self.model.beam_initial_state.name] = prev_state
        input_feed[self.model.beam_key_states.name] = key_states
        input_feed[self.model.key_mask.name] = key_mask
        input_feed[self.model.beam_global_trace.name] = global_trace
        input_feed[self.model.beam_topic_trace.name] = topic_trace
        input_feed[self.model.beam_his_mem.name] = his_mem
        input_feed[self.model.beam_his_mem_mask.name] = his_mem_mask

        output_feed = [self.model.next_out, self.model.next_state, self.model.next_align]

        [next_output, next_state, next_align] = self.sess.run(output_feed, input_feed)

        return next_output, next_state, next_align

    def his_mem_computer(self, his_mem, enc_outs, write_mask, global_trace):
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        input_feed[self.model.gama] = [0.05]
        input_feed[self.model.beam_his_mem.name] = his_mem

        for l in range(self.enc_len):
            input_feed[self.model.beam_enc_outs[l]] = enc_outs[l]
            input_feed[self.model.write_masks[0][l]] =write_mask[l]
        input_feed[self.model.beam_global_trace.name] = global_trace

        output_feed = [self.model.new_his_mem] 
        [new_his_mem] = self.sess.run(output_feed, input_feed)

        return new_his_mem

    def topic_trace_computer(self, key_states, prev_topic_trace, key_align):
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        input_feed[self.model.beam_key_states.name] = key_states
        input_feed[self.model.beam_topic_trace.name] = prev_topic_trace
        input_feed[self.model.beam_key_align.name] = key_align
        output_feed = [self.model.new_topic_trace]

        [new_topic_trace] = self.sess.run(output_feed, input_feed)
        return new_topic_trace

    def global_trace_computer(self, prev_global_trace, prev_attn_states):
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        input_feed [self.model.beam_global_trace] = prev_global_trace
        input_feed[self.model.beam_attn_states] = prev_attn_states
        output_feed = [self.model.new_global_trace]

        [new_global_trace] = self.sess.run(output_feed, input_feed)
        return new_global_trace


if __name__ == "__main__":
    tf.app.run()