import pickle
import numpy as np
import random
import copy

class PoetryTool(object):
    def __init__(self, sens_num, 
        key_slots, enc_len, dec_len):
        self.__sens_num = sens_num
        self.__key_slots = key_slots
        self.__enc_len = enc_len
        self.__dec_len = dec_len
    
    def line2chars(self, line):
        '''Split line to characters and save as list'''
        line = line.strip()
        line = line.replace('UNK', 'U')
        chars = [c for c in line]
        return chars

    def chars2idxes(self, chars):
        ''' Characters to idx list '''
        idxes = []
        for c in chars:
            if c in self.__vocab:
                idxes.append(self.__vocab[c])
            else:
                idxes.append(self.__vocab['UNK'])
        return idxes

    def idxes2chars(self, idxes, omit_special=True):
        chars = []
        for idx in idxes:
            if  (idx == self.__PAD_ID or idx == self.__B_ID 
                or idx == self.__E_ID or idx == self.__M_ID) and omit_special:
                continue
            chars.append(self.__ivocab[idx])

        return chars

    def get_vocab(self):
        return copy.deepcopy(self.__vocab)

    def get_ivocab(self):
        return copy.deepcopy(self.__ivocab)

    def get_vocab_size(self):
        if self.__vocab:
            return len(self.__vocab)
        else:
            return -1

    def get_PAD_ID(self):
        assert self.__vocab is not None
        return self.__PAD_ID

    def get_special_IDs(self):
        assert self.__vocab is not None
        return self.__PAD_ID, self.__UNK_ID, self.__B_ID, self.__E_ID, self.__M_ID

    def greedy_search(self, outputs):
        outidx = [int(np.argmax(logit, axis=0)) for logit in outputs]
        #print (outidx)
        if self.__E_ID in outidx:
            outidx = outidx[:outidx.index(self.__E_ID)]

        chars = self.idxes2chars(outidx)
        sentence = "".join(chars)
        return sentence

    def load_dic(self, vocab_path, ivocab_path):
        vocab_file = open(vocab_path, 'rb')
        dic = pickle.load(vocab_file)
        vocab_file.close()

        ivocab_file = open(ivocab_path, 'rb')
        idic = pickle.load(ivocab_file)
        ivocab_file.close()

        self.__vocab = dic
        self.__ivocab = idic

        self.__PAD_ID = dic['PAD']
        self.__UNK_ID = dic['UNK']
        self.__E_ID = dic['<E>']
        self.__B_ID = dic['<B>']
        self.__M_ID = dic['<M>']

    def build_fvec(self):
        print ("build function chars...")
        fin = open("data/other/fchars.txt", 'r')
        lines = fin.readlines()
        fin.close()
        chars = [line.strip() for line in lines]
        idxes = self.chars2idxes(chars)
        print ("f char num:%d" % (len(idxes)))
        return idxes

    def build_data(self, data_path, batch_size, mode):
        '''
        Build data as batches.
        NOTE: Please run load_dic() at first.
        '''
        corpus_file = open(data_path, 'rb')
        corpus = pickle.load(corpus_file)
        corpus_file.close()

        # TMP
        corpus = corpus[0:500]

        if mode == 'train':
            batches, batch_num = self.build_batches(corpus, batch_size) 
        elif mode == 'pre_train':
            batches, batch_num = self.build_ae_batches(corpus, batch_size) 

        return batches, batch_num

    def build_ae_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))  
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size - len(instances))

            # Build all batched data
            data_dic = {}

            poems = [instance[1] for instance in instances] # All poems
            genre_patterns = [instance[2] for instance in instances]
            for i in range(0, self.__sens_num-1):
                line0 = [poem[i] for poem in poems]
                line1 = [poem[i+1] for poem in poems]
                phs = [pattern[i+1] for pattern in genre_patterns]

                batch_enc_inps, batch_dec_inps, batch_weights, enc_mask, len_inps, \
                    ph_inps, batch_write_mask = self.get_batch_sentence(line0, line1, phs, batch_size)

                data_dic['enc_inps'] = batch_enc_inps
                data_dic['dec_inps'] = batch_dec_inps
                data_dic['trg_weights'] = batch_weights
                data_dic['ph_inps'] = ph_inps
                data_dic['len_inps'] = len_inps

                batched_data.append(data_dic)

        random.shuffle(batched_data)
        batch_num = len(batched_data)
        return batched_data, batch_num

    def build_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))  
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size - len(instances))

            # Build all batched data
            data_dic = {}
            all_enc_inps, all_dec_inps, all_trg_weights = [], [], []
            all_enc_mask, all_write_mask = [], []
            all_len_inps, all_ph_inps = [], []

            poems = [instance[1] for instance in instances] # All poems
            genre_patterns = [instance[2] for instance in instances]
            for i in range(-1, self.__sens_num-1):
                if i < 0:
                    line0 = [[] for poem in poems]
                else:
                    line0 = [poem[i] for poem in poems]
                
                line1 = [poem[i+1] for poem in poems]
                phs = [pattern[i+1] for pattern in genre_patterns]

                batch_enc_inps, batch_dec_inps, batch_weights, enc_mask, len_inps, \
                    ph_inps, batch_write_mask = self.get_batch_sentence(line0, line1, phs, batch_size)

                all_enc_inps.append(batch_enc_inps)
                all_dec_inps.append(batch_dec_inps)
                all_trg_weights.append(batch_weights)
                all_enc_mask.append(enc_mask)
                all_len_inps.append(len_inps)
                all_ph_inps.append(ph_inps)
                all_write_mask.append(batch_write_mask)

            data_dic['enc_inps'] = all_enc_inps
            data_dic['dec_inps'] = all_dec_inps
            data_dic['trg_weights'] = all_trg_weights
            data_dic['enc_mask'] = all_enc_mask
            data_dic['ph_inps'] = all_ph_inps
            data_dic['len_inps'] = all_len_inps
            data_dic['write_masks'] = all_write_mask

            # Build key batch
            keysvec = [instance[0] for instance in instances]
            key_inps = [[] for x in range(self.__key_slots)]
            key_mask = []
            
            for i in range(0, batch_size):
                keys = keysvec[i] # Batch_size * at most 4
                mask = [1.0]*len(keys) + [0.0]*(self.__key_slots-len(keys))
                mask = np.reshape(mask, [self.__key_slots, 1])
                key_mask.append(mask)
                for step in range(0, len(keys)):
                    key = keys[step]
                    assert len(key) <= 2
                    key_inps[step].append(key + [self.__PAD_ID] * (2-len(key)))
                for step in range(0, self.__key_slots-len(keys)):
                    key_inps[len(keys)+step].append([self.__PAD_ID] * 2)

            # key_inputs: key_slots, [id1, id2], batch_size
            batch_key_inps = [[] for x in range(self.__key_slots)]
            for step in range(0, self.__key_slots):
                # NOTE: Each topic word must consist of no more than 2 characters
                batch_key_inps[step].append(np.array([key_inps[step][i][0] for i in range(batch_size)]))
                batch_key_inps[step].append(np.array([key_inps[step][i][1] for i in range(batch_size)]))

            key_mask = np.array(key_mask)
            data_dic['key_mask'] = key_mask
            data_dic['key_inps'] = batch_key_inps

            batched_data.append(data_dic)

        return batched_data, batch_num

    def get_batch_sentence(self, inputs, outputs, phs, batch_size):
        assert len(inputs) == len(outputs) == len(phs) == batch_size
        enc_inps, dec_inps = [], []
        ph_inps, len_inps = [], []
        enc_mask, write_mask = [], []

        for i in range(batch_size):
            enc_inp = inputs[i]
            dec_inp = outputs[i] + [self.__E_ID]
            ph = phs[i]

            # Encoder inputs are padded
            enc_pad_size = self.__enc_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)
            mask = [1.0] * (len(enc_inp)) + [0.0] * (enc_pad_size)
            mask = np.reshape(mask, [self.__enc_len, 1])
            enc_mask.append(mask)
            write_mask.append(mask)

            # Decoder inputs get an extra "<B>" symbol, and are padded then.
            dec_pad_size = self.__dec_len - len(dec_inp) - 1
            dec_inps.append([self.__B_ID] + dec_inp + [self.__PAD_ID] * dec_pad_size)

            ph_inps.append(ph+[0]*(self.__dec_len-len(ph)))
            len_inp = list(range(len(dec_inp)+1, 0, -1)) + [0]*(dec_pad_size)
            len_inps.append(len_inp)

        # Create batch-major vectors.
        batch_enc_inps, batch_dec_inps, batch_weights = [], [], []
        batch_ph_inps, batch_write_mask = [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(self.__enc_len):
            batch_enc_inps.append(np.array([enc_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))
            batch_ph_inps.append(np.array([ph_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))
            batch_write_mask.append(np.array([write_mask[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

        for length_idx in range(self.__dec_len):
            batch_dec_inps.append(np.array([dec_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in range(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < self.__dec_len - 1:
                    target = dec_inps[batch_idx][length_idx + 1]
                if length_idx == self.__dec_len - 1 or target == self.__PAD_ID:
                    batch_weight[batch_idx] = 0.0

            batch_weights.append(batch_weight)

        #
        enc_mask = np.array(enc_mask)
        len_inps = np.transpose(np.array(len_inps))
        ph_inps = np.transpose(np.array(ph_inps))
        return batch_enc_inps, batch_dec_inps, batch_weights, \
            enc_mask, len_inps, ph_inps, batch_write_mask

    # -----------------------------------------------------------
    # Tools for beam search
    def beam_get_sentence(self, idxes):
        if idxes is not list:
            idxes = list(idxes)
        if self.__E_ID in idxes:
          idxes = idxes[:idxes.index(self.__E_ID)]

        chars = self.idxes2chars(idxes)
        sentence = "".join(chars)

        return sentence

    def gen_enc_beam(self, sentence, trg_len, batch_size):
        enc_inps, len_inps, enc_mask = [], [], []
        trg_len += 1

        for i in range(0, batch_size):
            enc_inp = sentence
            enc_pad_size = self.__enc_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)
            mask = [1.0] * (len(enc_inp)) + [0.0] * (enc_pad_size)
            mask = np.reshape(mask, [self.__enc_len,1])
            enc_mask.append(mask)

            dec_pad_size = self.__dec_len - trg_len - 1
            len_inp = list(range(trg_len+1, 0, -1)) + [0]*(dec_pad_size)
            len_inps.append(len_inp)


        batch_enc_inps= []
        for length_idx in range(self.__enc_len):
            batch_enc_inps.append(
                np.array([enc_inps[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))
        
        len_inps = np.transpose(np.array(len_inps))
        enc_mask = np.array(enc_mask)
        return batch_enc_inps, enc_mask, len_inps

    
    def gen_batch_key_beam(self, keys, batch_size):
        ''' Generate batch input for batch keywords '''
        keysvec = [keys for _ in range(0, batch_size)]
        key_inps = [[] for x in range(self.__key_slots)]
        key_mask = []
        
        for i in range(0, batch_size):
            keys = keysvec[i] # batch_size * at most 4
            mask = [1.0]*len(keys) + [0.0]*(self.__key_slots-len(keys))
            mask = np.reshape(mask, [self.__key_slots, 1])
            key_mask.append(mask)
            for step in range(0, len(keys)):
                key = keys[step]
                key_inps[step].append(key + [self.__PAD_ID] * (2-len(key)))
            for step in range(0, self.__key_slots-len(keys)):
                key_inps[len(keys)+step].append([self.__PAD_ID] * 2)

        # key_inputs: key_slots, [id1, id2], batch_size
        batch_key_inps = [[] for x in range(self.__key_slots)]
        for step in range(0, self.__key_slots):
            batch_key_inps[step].append(np.array([key_inps[step][i][0] for i in range(batch_size)]))
            batch_key_inps[step].append(np.array([key_inps[step][i][1] for i in range(batch_size)]))
        key_mask = np.array(key_mask)

        return batch_key_inps, key_mask