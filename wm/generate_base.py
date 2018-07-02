from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from model import PoemModel
from tool import PoetryTool
from DataTool import data_tool
from config import hps
import numpy as np

class Generator(object):

    def __init__(self, beam_size, model_file=None):
        # Construct hyper-parameter
        self.hps = hps
        self.dtool = data_tool
        self.beam_size = beam_size
        self.tool = PoetryTool(sens_num=hps.sens_num,
            key_slots=hps.key_slots, enc_len=hps.bucket[0],
            dec_len=hps.bucket[1])
        if hps.init_emb == '':
            self.init_emb = None
        else:
            self.init_emb = np.load(self.hps.init_emb)
            print ("init_emb_size: %s" % str(np.shape(self.init_emb)))
        self.tool.load_dic(hps.vocab_path, hps.ivocab_path)

        vocab_size = self.tool.get_vocab_size()
        assert vocab_size > 0
        PAD_ID = self.tool.get_PAD_ID()
        assert PAD_ID > 0

        self.hps = self.hps._replace(vocab_size=vocab_size, 
            PAD_ID=PAD_ID, batch_size=beam_size, mode='decode')
        self.model = PoemModel(self.hps)

        self.EOS_ID, self.PAD_ID, self.GO_ID, self.UNK_ID \
            = self.tool.get_special_IDs()

        self.enc_len = self.hps.bucket[0]
        self.dec_len = self.hps.bucket[1]
        self.topic_trace_size = self.hps.topic_trace_size
        self.key_slots = self.hps.key_slots
        self.his_mem_slots = self.hps.his_mem_slots
        self.his_mem_size = self.hps.his_mem_size
        self.global_trace_size = self.hps.global_trace_size
        self.hidden_size = self.hps.hidden_size

        self.sess = tf.InteractiveSession()

        if model_file is None:
            self.load_model(self.sess, self.model)
        else:
            self.load_model_by_path(self.sess, self.model, model_file)


        self.__buildPH()

    def load_model(self, session, model):
        """load parameters in session."""
        ckpt = tf.train.get_checkpoint_state(self.hps.model_path)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)

    def load_model_by_path(self, session, model, modefile):
        """load parameters in session."""
        if tf.gfile.Exists(modefile):
            print("Reading model parameters from %s" %modefile)
            model.saver.restore(session, modefile)
        else:
            raise ValueError("%s not found! " % modefile)

    def __buildPH(self):
        self.__PHDic = self.dtool.buildPHDicForIdx(
            copy.deepcopy(self.tool.get_vocab()))
        
    def addtionFilter(self, trans, pos):
        pos -= 1
        preidx = range(0, pos)
        batch_size = len(trans)
        forbidden_list = [[] for _ in xrange(0, batch_size)]

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

    def beam_search(self, sess, sen, len_inps, ori_key_states, key_initial_state, ori_topic_trace, 
        ori_his_mem, ori_his_mem_mask, ori_global_trace, enc_mask, ori_key_mask, repeatidxvec, phs):
        trg_length = len(phs)        
        beam_size = self.beam_size
        n_samples = beam_size

        enc_state, attn_states = self.model.encoder_computer(sess, sen, enc_mask)
        enc_states = copy.deepcopy(attn_states)
        enc_mask = np.array(enc_mask)

        key_states = copy.deepcopy(ori_key_states)
        topic_trace= copy.deepcopy(ori_topic_trace)
        global_trace = copy.deepcopy(ori_global_trace)
        his_mem = copy.deepcopy(ori_his_mem)
        his_mem_mask = copy.deepcopy(ori_his_mem_mask)
        key_mask = copy.deepcopy(ori_key_mask)

        fin_trans, fin_costs, fin_align = [], [], []

        trans = [[] for i in xrange(0, beam_size)]
        costs = [0.0]

        key_align = []
        for i in range(beam_size):
            key_align.append(np.zeros([1, self.key_slots], dtype=np.float32))

        state = enc_state
        if not (key_initial_state is None):
            state = key_initial_state
        inp = np.array([self.GO_ID]*beam_size)

        ph_inp = [phs[0]]*n_samples

        output, state, alignments = self.model.decoder_state_computer(sess, inp,
            len_inps[0], ph_inp, state, attn_states, key_states, his_mem, global_trace,
            enc_mask, key_mask, his_mem_mask, topic_trace)

        for k in range(1, 2*trg_length):
            if n_samples == 0:
                break

            if k == 1:
                output = output[0, :]
          
            log_probs = np.log(output)
            next_costs = np.array(costs)[:, None] - log_probs

            # Form a beam for the next iteration
            new_trans = [[] for i in xrange(0, n_samples)]
            new_costs = np.zeros(n_samples, dtype="float32")
            new_states = np.zeros((n_samples, self.hidden_size), dtype="float32")
            new_align = [[] for i in xrange(0, n_samples)]

            inputs = np.zeros(n_samples, dtype="int64")
            
            # Note that here k < len(gls) means that we don't put hard constraint on yun
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
                current_align = alignments[orig_idx, align_start:align_end]
                new_align[i] = np.concatenate((key_align[orig_idx], [current_align]), axis=0)
                new_states[i] = state[orig_idx, :]
                inputs[i] = next_word

            # Filter the sequences that end with end-of-sequence character
            trans, costs, indices, key_align = [], [], [], []

            for i in range(n_samples):
                if new_trans[i][-1] != self.EOS_ID:
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

            output, state, alignments = self.model.decoder_state_computer(sess, inputs, specify_len,
                ph_inp, new_states, attn_states, key_states, his_mem, global_trace, enc_mask, key_mask,
                his_mem_mask,  topic_trace)
            

        #print (np.shape(fin_align))
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

        return fin_trans, fin_costs, fin_align, enc_states

    def get_new_global_trace(self, sess, history, ori_enc_states, beam_size):

        enc_states = np.expand_dims(ori_enc_states, axis=0)
        prev_history = np.expand_dims(history[0, :], 0)
        #print (np.shape(prev_encoder_state))
        #tt = input(">")
        new_history = self.model.global_trace_computer(sess, prev_history, enc_states)
        new_history = np.tile(new_history, [beam_size, 1])
        return new_history

    def get_new_his_mem(self, sess, ori_his_mem, enc_states, ori_global_trace, beam_size, src_len):
        his_mem = np.expand_dims(ori_his_mem, axis=0)
        fin_states = []
        for i in xrange(0, np.shape(enc_states)[0]):
            fin_states.append(np.expand_dims(enc_states[i], 0))

        mask = [np.ones((1, 1))] * src_len + [np.zeros((1, 1))] * (np.shape(enc_states)[0]-src_len)
        global_trace = np.expand_dims(ori_global_trace[0, :], 0)
        new_his_mem = self.model.his_mem_computer(sess, his_mem, 
            fin_states, mask, global_trace)

        new_his_mem = np.tile(new_his_mem, [beam_size, 1, 1])
        return new_his_mem

    def get_new_topic_trace(self, sess, ori_topic_trace, key_align, ori_key_states, beam_size):
        key_states = np.expand_dims(ori_key_states[0, :, :], 0)
        topic_trace = np.expand_dims(ori_topic_trace, axis=0)
        key_align = np.mean(key_align, axis=0) 
        key_align = np.expand_dims(key_align, axis=0)
        new_topic_trace = self.model.topic_trace_computer(sess, 
            key_states, topic_trace, key_align)
        new_topic_trace = np.tile(new_topic_trace, [beam_size, 1])
        return new_topic_trace

    def generate_one(self, keystr, pattern):
        beam_size = self.beam_size
        sens_num = len(pattern)
        keys = keystr.strip()
        ans, repeatidxes = [], []
        print ("using keywords: %s" % (keystr))
        keys = keystr.split(" ")
        keys_idxes = [self.tool.chars2idxes(self.tool.line2chars(key)) for key in keys]
        #print (keys_idxes)
        key_inps, key_mask = self.tool.gen_batch_key_beam(keys_idxes, beam_size)

        # Calculate initial_key state and key_states
        key_initial_state, key_states = self.model.key_memory_computer(self.sess, key_inps, key_mask)

        his_mem_mask = np.zeros([beam_size, self.his_mem_slots], dtype=np.float32)
        global_trace = np.zeros([beam_size, self.global_trace_size], dtype='float32')
        his_mem = np.zeros([beam_size, self.his_mem_slots, self.his_mem_size], dtype='float32')
        topic_trace = np.zeros([beam_size, self.topic_trace_size+self.key_slots], dtype='float32')
        
        # Generate the first line, line0 is an empty list
        sen = []
        for step in xrange(0, sens_num):
            print("generating %d line..." % (step+1))
            phs = pattern[step]
            trg_len = len(phs)
            if step > 0:
                key_initial_state = None
            src_len = len(sen)
            batch_sen, enc_mask, len_inps = self.tool.gen_batch_beam(sen, trg_len, beam_size)
            trans, costs, align, enc_states = self.beam_search(self.sess, batch_sen, len_inps, key_states,
                key_initial_state, topic_trace, his_mem, his_mem_mask, global_trace, enc_mask, key_mask,
                repeatidxes, phs)

            trans, costs, align, enc_states = self.pFilter(trans, costs, align, enc_states, trg_len)
            
            if len(trans) == 0:
                return [], ("line %d generation failed!" % (step+1))

            which = 0

            his_mem = self.get_new_his_mem(self.sess, his_mem[which, :, :], 
                enc_states[which], global_trace, beam_size, src_len)

            if step >= 1:
                one_his_mem = his_mem[which, :, :]
                his_mem_mask = np.sum(np.abs(one_his_mem), axis=1)
                his_mem_mask = his_mem_mask != 0
                his_mem_mask = np.tile(his_mem_mask.astype(np.float32), [beam_size, 1])

            sentence = self.tool.beam_get_sentence(trans[which])
            sentence = sentence.strip()
            ans.append(sentence)
            attn_aligns = align[which][0:trg_len, :]
            topic_trace = self.get_new_topic_trace(self.sess, topic_trace[which, :], attn_aligns,
                key_states, beam_size)
            global_trace = self.get_new_global_trace(self.sess, global_trace, enc_states[which], beam_size)

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

def main(_):
    pass

if __name__ == "__main__":
    tf.app.run()
