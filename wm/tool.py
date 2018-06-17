import cPickle
import numpy as np
import random
import copy


class DataTool(object):
    """docstring for Tool"""
    def __init__(self, sens_num, 
        key_slots, enc_len, dec_len):
        self.__sens_num = sens_num
        self.__key_slots = key_slots
        self.__enc_len = enc_len
        self.__dec_len = dec_len

    
    def line2chars(self, line):
        '''Split line to characters and save as list'''
        line = line.decode("utf-8")
        chars = []
        for c in line:
            chars.append(c.encode("utf-8"))
        return chars

    def chars2idxes(self, chars):
        ''' Characters to idx list '''
        idxes = []
        for c in chars:
            if c in self.vocab:
                idxes.append(self.__vocab[c])
            else:
                idxes.append(self.__vocab['UNK'])
        return idxes

    def idxes2chars(self, idxes):
        chars = []
        for idx in idxes:
            if  (idx == self.__PAD_ID or idx == self.__GO_ID 
                or idx == self.__EOS_ID):
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
        if self.__vocab:
            return self.__PAD_ID
        else:
            return -1

    def get_special_IDs(self):
        return self.__EOS_ID, self.__PAD_ID, self.__GO_ID, self.__UNK_ID

    def greedy_search(self, outputs):
        outidx = [int(np.argmax(logit, axis=0)) for logit in outputs]
        #print (outidx)
        if self.EOS_ID in outidx:
            outidx = outidx[:outidx.index(self.EOS_ID)]

        sentence = self.idxes2sen(outidx)
        sentence = " ".join(sentence)
        return sentence

    def load_dic(self, file_dir, if_return=False):
        vocab_file = open(file_dir + '/vocab.pkl', 'rb')
        dic = cPickle.load(vocab_file)
        vocab_file.close()

        ivocab_file = open(file_dir + '/ivocab.pkl', 'rb')
        idic = cPickle.load(ivocab_file)
        ivocab_file.close()

        self.__vocab = dic
        self.__ivocab = idic

        self.__EOS_ID = dic['</S>']
        self.__PAD_ID = dic['PAD']
        self.__GO_ID = dic['GO']
        self.__UNK_ID = dic['UNK']

        if if_return:
            return dic, idic


    def load_data(self, file_dir, trainfile, validfile):
        '''
        loading  training data, including vocab, inverting vocab and corpus
        '''
        if not self.vocab:
            self.load_dic(file_dir)    

        corpus_file = open(file_dir + '/' + trainfile, 'rb')
        train_corpus = cPickle.load(corpus_file)
        corpus_file.close()

        corpus_file = open(file_dir + '/' + validfile, 'rb')
        valid_corpus = cPickle.load(corpus_file)
        corpus_file.close()

        return train_corpus, valid_corpus

    def build_data(self, file_dir, batch_size, trainfile, validfile):
        train_data, valid_data = \
            self.load_data(file_dir, trainfile, validfile)

        #TMP
        '''
        train_data = train_data[0:1000]
        valid_data = valid_data[0:200]
        '''
        train_batches, train_batch_num = self.build_batches(
            train_data, batch_size) 
        valid_batches, valid_batch_num = self.build_batches(
            valid_data, batch_size)

        return train_batch_num, \
            valid_batch_num, train_batches, valid_batches

    def build_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / batch_size))

        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))
            data_dic = {}
            all_dec_inps = []
            all_trg_weights = []
            poems = [instance[1] for instance in instances] # all poems
            for i in xrange(0, self.sens_num):
                lines = [poem[i] for poem in poems]
                batch_dec_inps, batch_weights, len_inps = self.get_batch_sentence(lines, batch_size)
                all_dec_inps.append(batch_dec_inps)
                all_trg_weights.append(batch_weights)

            data_dic['decoder_inputs'] = all_dec_inps
            data_dic['target_weights'] = all_trg_weights
            data_dic['len_inputs'] = len_inps

            # build key batch
            keysvec = [instance[0] for instance in instances]
            key_inputs = [[] for x in xrange(self.key_slots)]
            key_mask = []
                
            for i in range(0, batch_size):
                keys = keysvec[i] # batch_size * at most 4
                mask = [1.0]*len(keys) + [0.0]*(self.key_slots-len(keys))
                mask = np.reshape(mask, [self.key_slots, 1])
                key_mask.append(mask)
                for step in xrange(0, len(keys)):
                    key = keys[step]
                    key_inputs[step].append(key + [self.PAD_ID] * (2-len(key)))
                for step in xrange(0, self.key_slots-len(keys)):
                    key_inputs[len(keys)+step].append([self.PAD_ID] * 2)

            # key_inputs: key_slots, [id1, id2], batch_size
            batch_key_inputs = [[] for x in xrange(self.key_slots)]
            for step in xrange(0, self.key_slots):
                batch_key_inputs[step].append(np.array([key_inputs[step][i][0] for i in xrange(batch_size)]))
                batch_key_inputs[step].append(np.array([key_inputs[step][i][1] for i in xrange(batch_size)]))

            key_mask = np.array(key_mask)
            data_dic['key_mask'] = key_mask
            data_dic['key_inputs'] = batch_key_inputs
            batched_data.append(data_dic)

        return batched_data, batch_num

    def get_batch_sentence(self, outputs, batch_size):
        assert  len(outputs) == batch_size
        decoder_size = self.sen_len
        decoder_inputs = []
        len_inputs = []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in xrange(batch_size):
            yan = len(outputs[i])
            #print (outputs[i])
            assert yan == 5 or yan == 7
            if yan == 5:
                len_inputs.append([0.0, 1.0])
            else:
                len_inputs.append([1.0, 0.0])

            decoder_input = outputs[i] + [self.EOS_ID]

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([self.GO_ID] + decoder_input +
                                  [self.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_decoder_inputs, batch_weights = [], []

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1
                # forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == self.PAD_ID:
                    batch_weight[batch_idx] = 0.0

            batch_weights.append(batch_weight)

        return batch_decoder_inputs, batch_weights, len_inputs

    ''' generate batch input for batch keywords '''
    def build_batch_key_beam(self, keys, batch_size = 10):
        # build key batch
        keysvec = [keys for _ in xrange(0, batch_size)]
        key_inputs = [[] for x in xrange(self.key_slots)]
        key_mask = []
        
        for i in range(0, batch_size):
            keys = keysvec[i] # batch_size * at most 4
            mask = [1.0]*len(keys) + [0.0]*(self.key_slots-len(keys))
            mask = np.reshape(mask, [self.key_slots, 1])
            key_mask.append(mask)
            for step in xrange(0, len(keys)):
                key = keys[step]
                key_inputs[step].append(key + [self.PAD_ID] * (2-len(key)))
            for step in xrange(0, self.key_slots-len(keys)):
                key_inputs[len(keys)+step].append([self.PAD_ID] * 2)

        # key_inputs: key_slots, [id1, id2], batch_size
        batch_key_inputs = [[] for x in xrange(self.key_slots)]
        for step in xrange(0, self.key_slots):
            batch_key_inputs[step].append(np.array([key_inputs[step][i][0]
                for i in xrange(batch_size)]))
            batch_key_inputs[step].append(np.array([key_inputs[step][i][1]
                for i in xrange(batch_size)]))
        key_mask = np.array(key_mask)

        return batch_key_inputs, key_mask

    def get_batch_trg(self, trg):
        decoder_input = trg + [self.EOS_ID]
        decoder_pad_size = self.sen_len - len(decoder_input) -1
        fin_trg = [self.GO_ID] + decoder_input + [self.PAD_ID] * decoder_pad_size

        weights = []
        for length_idx in xrange(self.sen_len):
            weight = 1.0
            if length_idx < self.sen_len - 1:
                target = fin_trg[length_idx + 1]
            if length_idx == self.sen_len - 1 or target == self.PAD_ID:
                weight = 0.0
            weights.append(weight)

        return fin_trg, weights
