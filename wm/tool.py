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
        if self.__EOS_ID in outidx:
            outidx = outidx[:outidx.index(self.__EOS_ID)]

        chars = self.idxes2chars(outidx)
        sentence = " ".join(chars)
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
        if not self.__vocab:
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
        batch_num = int(np.ceil(len(data) / float(batch_size)))  
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size - len(instances))

            # generate all batch data
            data_dic = {}
            all_enc_inps, all_dec_inps, all_trg_weights = [], [], []
            all_enc_mask, all_write_mask = [], []
            all_len_inps, all_ph_inps = [], []

            # build sentence batch
            poems = [instance[1] for instance in instances] # all poems
            genre_patterns = [instance[2] for instance in instances]
            for i in xrange(-1, self.__sens_num-1):
                if i <0:
                    line0 = [[] for poem in poems]
                else:
                    line0 = [poem[i] for poem in poems]
                
                line1 = [poem[i+1] for poem in poems]
                gls = [pattern[i+1] for pattern in genre_patterns]

                batch_enc_inps, batch_dec_inps, batch_weights, enc_mask, len_inps, \
                    ph_inps, batch_write_mask = self.get_batch_sentence(line0, line1, gls, batch_size)

                all_enc_inps.append(batch_enc_inps)
                all_dec_inps.append(batch_dec_inps)
                all_trg_weights.append(batch_weights)
                all_enc_mask.append(enc_mask)
                all_len_inps.append(len_inps)
                all_ph_inps.append(ph_inps)
                all_write_mask.append(batch_write_mask)

            #print (np.shape(all_gl_inputs))

            data_dic['enc_inps'] = all_enc_inps
            data_dic['dec_inps'] = all_dec_inps
            data_dic['trg_weights'] = all_trg_weights
            data_dic['enc_mask'] = all_enc_mask
            data_dic['ph_inps'] = all_ph_inps
            data_dic['len_inps'] = all_len_inps
            data_dic['write_masks'] = all_write_mask

            # build key batch
            keysvec = [instance[0] for instance in instances]
            key_inputs = [[] for x in xrange(self.hps.key_slots)]
            key_mask = []
            
            for i in range(0, batch_size):
                keys = keysvec[i] # batch_size * at most 4
                mask = [1.0]*len(keys) + [0.0]*(self.hps.key_slots-len(keys))
                mask = np.reshape(mask, [self.hps.key_slots, 1])
                key_mask.append(mask)
                for step in xrange(0, len(keys)):
                    key = keys[step]
                    key_inputs[step].append(key + [self.PAD_ID] * (2-len(key)))
                for step in xrange(0, self.hps.key_slots-len(keys)):
                    key_inputs[len(keys)+step].append([self.PAD_ID] * 2)

            # key_inputs: key_slots, [id1, id2], batch_size
            batch_key_inputs = [[] for x in xrange(self.hps.key_slots)]
            for step in xrange(0, self.hps.key_slots):
                batch_key_inputs[step].append(np.array([key_inputs[step][i][0] for i in xrange(batch_size)]))
                batch_key_inputs[step].append(np.array([key_inputs[step][i][1] for i in xrange(batch_size)]))

            key_mask = np.array(key_mask)

            data_dic['key_mask'] = key_mask
            data_dic['key_inputs'] = batch_key_inputs

            batched_data.append(data_dic)

        return batched_data, batch_num

    def get_batch_sentence(self, inputs, outputs, gls, batch_size, bucket_id=0):
        assert len(inputs) == len(outputs) == batch_size
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        gl_inputs = []
        encoder_mask = []
        len_inputs = []
        write_mask = []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in xrange(batch_size):
            encoder_input = inputs[i]
            decoder_input = outputs[i] + [self.EOS_ID]
            gl = gls[i]

            # Encoder inputs are padded and then reversed.
            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_pad = [self.PAD_ID] * encoder_pad_size
            encoder_inputs.append(encoder_input + encoder_pad)
            mask = [1.0] * (len(encoder_input)) + [0.0] * (encoder_pad_size)
            mask = np.reshape(mask, [encoder_size, 1])
            encoder_mask.append(mask)
            write_mask.append(mask)

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([self.GO_ID] + decoder_input +
                                  [self.PAD_ID] * decoder_pad_size)

            #print (decoder_inputs[-1])
            #print (" ".join([self.ivocab[wid] for wid in decoder_inputs[-1]]))
            gl_inputs.append(gl+[0]*(decoder_size-len(gl)))
            #print (gl_inputs[-1])


            len_input = range(len(decoder_input)+1, 0, -1) + [0]*(decoder_pad_size)
            #print(len_input)
            #tt = input(">")
            len_inputs.append(len_input)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        batch_gl_inputs = []
        batch_write_mask = []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                                  for batch_idx in xrange(batch_size)], dtype=np.int32))
            batch_gl_inputs.append(np.array([gl_inputs[batch_idx][length_idx]
                                                  for batch_idx in xrange(batch_size)], dtype=np.int32))
            batch_write_mask.append(np.array([write_mask[batch_idx][length_idx]
                                                  for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create
        # weights.
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

        #
        encoder_mask = np.array(encoder_mask)
        len_inputs = np.transpose(np.array(len_inputs))
        gl_inputs = np.transpose(np.array(gl_inputs))
        #print ("_____________")
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_mask, len_inputs, gl_inputs, batch_write_mask


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
