import numpy as np

'''
The tool class for poetry project
'''
class PoetryTool(object):
    def __init__(self):
        pass

    '''
    split line to character and save as list
    '''
    def lineSplit2list(self, line):
        sentence = []
        for i in range(0, len(line), 3):
            c = line[i:i + 3]
            sentence.append(c)
        return sentence

    """Compute softmax values for each sets of scores in x."""
    ''' Each line of x is a data line '''
    def softmax(self, x):
        ans = np.zeros(np.shape(x), dtype=np.float32)
        for i in range(np.shape(x)[0]):
            c = x[i, :]
            ans[i, :] = np.exp(c) / np.sum(np.exp(c), axis=0)

        return ans

    ''' normalize each line of matrix '''
    def norm_matrxi(self, matrix):
        l = np.shape(matrix)[0]
        for i in range(0, l):
            if np.sum(matrix[i, :]) == 0:
                matrix[i, :] = 0
            else:
                matrix[i, :] = matrix[i, :] / np.sum(matrix[i, :])
        return matrix

    '''
    change id to a sentence
    ws: if use white space to split chracters
    '''
    def beam_get_sentence(self, idxes, ivocab, EOS_ID, ws=False):
        if idxes is not list:
            idxes = list(idxes)
        if EOS_ID in idxes:
          idxes = idxes[:idxes.index(EOS_ID)]

        sentence = self.idxes2senlist(idxes, ivocab)
        if ws:
            sentence = " ".join(sentence)
        else:
            sentence = "".join(sentence)
        return sentence

    ''' idxes to character list '''
    def idxes2senlist(self, idxes, idic):
        sentence = []
        for idx in idxes:
            if idx in idic:
                sentence.append(idic[idx])
            else:
                sentence.append('UNK')
        return sentence

    ''' character list to idx list '''
    def senvec2idxes(self, sentence, dic):
        idxes = []
        for c in sentence:
            if c in dic:
                idxes.append(dic[c])
            else:
                idxes.append(dic['UNK'])
        return idxes

    ''' generate batch input for encoder '''
    def gen_batch_beam(self, sentence, seq_len, trg_len, PAD_ID, batch_size = 10):
        inputs = [sentence] * batch_size
        encoder_inputs = []
        encoder_size = seq_len
        decoder_size = seq_len
        encoder_mask = []
        len_inputs = []
        trg_len += 1

        for i in xrange(len(inputs)):
            # in genearting, there is no EOS in the input sentence
            encoder_input = inputs[i]  

            # Encoder inputs are padded
            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_pad = [PAD_ID] * encoder_pad_size
            encoder_inputs.append(encoder_input + encoder_pad)
            mask = [1.0] * (len(encoder_input)) + [0.0] * (encoder_pad_size)
            mask = np.reshape(mask, [encoder_size,1])
            encoder_mask.append(mask)

            decoder_pad_size = decoder_size - trg_len - 1
            len_input = range(trg_len+1, 0, -1) + [0]*(decoder_pad_size)
            len_inputs.append(len_input)

        # create batch-major vectors from the data
        batch_encoder_inputs= []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))
        
        len_inputs = np.transpose( np.array(len_inputs))
        return batch_encoder_inputs, encoder_mask, len_inputs

    ''' generate batch input for batch keywords '''
    def gen_batch_key_beam(self, keys, key_slots, PAD_ID, batch_size = 10):
        # build key batch
        keysvec = [keys for _ in xrange(0, batch_size)]
        key_inputs = [[] for x in xrange(key_slots)]
        key_mask = []
        
        for i in range(0, batch_size):
            keys = keysvec[i] # batch_size * at most 4
            mask = [1.0]*len(keys) + [0.0]*(key_slots-len(keys))
            mask = np.reshape(mask, [key_slots, 1])
            key_mask.append(mask)
            for step in xrange(0, len(keys)):
                key = keys[step]
                key_inputs[step].append(key + [PAD_ID] * (2-len(key)))
            for step in xrange(0, key_slots-len(keys)):
                key_inputs[len(keys)+step].append([PAD_ID] * 2)

        # key_inputs: key_slots, [id1, id2], batch_size
        batch_key_inputs = [[] for x in xrange(key_slots)]
        for step in xrange(0, key_slots):
            batch_key_inputs[step].append(np.array([key_inputs[step][i][0] for i in xrange(batch_size)]))
            batch_key_inputs[step].append(np.array([key_inputs[step][i][1] for i in xrange(batch_size)]))
        key_mask = np.array(key_mask)

        return batch_key_inputs, key_mask 

    def get_batch_trg(self, trg, GO, EOS_ID, PAD_ID, sen_len):
        decoder_input = trg + [EOS_ID]
        decoder_pad_size = sen_len - len(decoder_input) -1
        fin_trg = [GO] + decoder_input + [PAD_ID] * decoder_pad_size

        weights = []
        for length_idx in xrange(sen_len):
            weight = 1.0
            if length_idx < sen_len - 1:
                target = fin_trg[length_idx + 1]
            if  length_idx == sen_len-1 or target == PAD_ID:
                weight = 0.0
            weights.append(weight)

        return fin_trg, weights

