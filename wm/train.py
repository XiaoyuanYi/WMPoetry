from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
import cPickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from model import PoemModel
from PoetryTool import PoetryTool
from tool import DataTool
from config import hps

class PoemTrainer(object):
    """construction for PoemTrainer"""

    def __init__(self):
        self.tool = PoetryTool()

        # Construct hyper-parameter
        self.hps = hps
        self.init_emb = np.load("data/train/init_emb.npy")
        print ("init_emb_size: %s" % str(np.shape(self.init_emb)))
        self.data_tool = DataTool(sens_num=self.hps.sens_num, 
            key_slots=self.hps.key_slots, enc_len=self.hps.bucket[0],
            dec_len=self.hps.bucket[1])
        self.data_tool.load_dic("data/train")

        vocab_size = self.data_tool.get_vocab_size()
        assert len(vocab_size) > 0
        PAD_ID = self.data_tool.get_PAD_ID()
        assert PAD_ID > 0

        self.hps = self.hps._replace(vocab_size=vocab_size, PAD_ID=PAD_ID)

        print("Params  sets: ")
        print (self.hps)
        print("___________________")
        t = input(">")


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

    def build_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / batch_size))  
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size - len(instances))

            # generate all batch data
            data_dic = {}
            all_encoder_inputs = []
            all_decoder_inputs = []
            all_target_weights = []
            all_encoder_mask = []
            all_len_inputs = []
            all_gl_inputs = []
            all_write_mask = []

            # build sentence batch
            poems = [instance[1] for instance in instances] # all poems
            glpatterns = [instance[2] for instance in instances]
            for i in xrange(-1, self.sens_num-1):
                if i <0:
                    line0 = [[] for poem in poems]
                else:
                    line0 = [poem[i] for poem in poems]
                
                line1 = [poem[i+1] for poem in poems]
                gls = [pattern[i+1] for pattern in glpatterns]

                batch_encoder_inputs, batch_decoder_inputs, batch_weights, encoder_mask, len_inputs, gl_inputs, batch_write_mask = self.get_batch_sentence(
                    line0, line1, gls, batch_size)

                all_encoder_inputs.append(batch_encoder_inputs)
                all_decoder_inputs.append(batch_decoder_inputs)
                all_target_weights.append(batch_weights)
                all_encoder_mask.append(encoder_mask)
                all_len_inputs.append(len_inputs)
                all_gl_inputs.append(gl_inputs)
                all_write_mask.append(batch_write_mask)

            #print (np.shape(all_gl_inputs))

            data_dic['encoder_inputs'] = all_encoder_inputs
            data_dic['decoder_inputs'] = all_decoder_inputs
            data_dic['target_weights'] = all_target_weights
            data_dic['encoder_mask'] = all_encoder_mask
            data_dic['gl_inputs'] = all_gl_inputs
            data_dic['len_inputs'] = all_len_inputs
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

    def create_model(self, session):
        """Create the model and initialize or load parameters in session."""
        model = PoemModel(self.hps)
        ckpt = tf.train.get_checkpoint_state("model")
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        return model

    '''
    search a output sentence by the simple greedy_decode
    '''

    def greedy_decode(self, outputs):
        outidx = [int(np.argmax(logit, axis=0)) for logit in outputs]
        #print (outidx)
        if self.EOS_ID in outidx:
            outidx = outidx[:outidx.index(self.EOS_ID)]

        sentence = self.idx2sentece(outidx)
        sentence = " ".join(sentence)
        return sentence

    def sample(self, encoder_inputs, decoder_inputs, key_inputs, outputs):

        sample_num = self.FLAGS.sample_num
        if sample_num > self.batch_size:
            sample_num = self.batch_size

        idxes = []  # Random select some examples
        while (len(idxes) < self.FLAGS.sample_num):
            which = np.random.randint(self.batch_size)
            if not which in idxes:
                idxes.append(which)

        #
        for idx in idxes:
            keys = []
            for i in xrange (0, self.hps.key_slots):
                key_idx = [key_inputs[i][0][idx], key_inputs[i][1][idx]]
                keys.append("".join(self.idx2sentece(key_idx)))
            key_str = " ".join(keys)

            # build original sentences
            print ("%s" % (key_str))
            for step in xrange(0, self.sens_num):
                inputs = [c[idx] for c in encoder_inputs[step]]
                inline = "".join(self.idx2sentece(inputs, True))

                target = [c[idx] for c in decoder_inputs[step]]
                tline = "".join(self.idx2sentece(target, True))

                outline = [c[idx] for c in outputs[step]]
                outline = self.greedy_decode(outline)

                print(inline.ljust(34) + " # " + tline.ljust(34) + " # " + outline.ljust(34) + " # ")
            print("__________________________\n")

    def run_validation(self, sess, model, valid_batches, valid_batch_num, epoch):
        print("run validation...")
        total_loss = 0.0
        for step in xrange(0, valid_batch_num):
            batch = valid_batches[step]
            outputs, step_loss  = model.step(sess, batch, True)
            total_loss += step_loss
        total_loss /= valid_batch_num
        info = "validation epoch: %d  loss: %.3f  ppl: %.2f" % (epoch, total_loss, np.exp(total_loss))
        print (info)
        fout = open("validlog.txt", 'a')
        fout.write(info + "\n")
        fout.close()

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Create model.
            model = self.create_model(sess)

            print (len((self.train_data)))
            print (len(self.valid_data))
            train_batches, train_batch_num = self.build_batches(self.train_data, self.batch_size) 
            valid_batches, valid_batch_num = self.build_batches(self.valid_data, self.batch_size)
            gama_decay_steps = int(train_batch_num // 20)
            print ("train_batch_num: %d" % (train_batch_num))
            print ("valid_batch_num: %d" % (valid_batch_num))
            print ("gama_decay_steps: %d" % (gama_decay_steps))

            for epoch in xrange(1, self.max_epoch+1):

                total_loss = 0.0
                time1 = time.time()
                for step in xrange(0, train_batch_num):
                    batch = train_batches[step]
                    #outputs, _, step_loss, l2_loss, debug, debug2 = model.step(sess, batch, False)
                    outputs, step_loss, l2_loss, (wb1, wb2, wb3, wb4, wb5), grads = model.step(sess, batch, False)
                    total_loss += step_loss

                    #print (step_loss, l2_loss)

                    if step % 2 == 0:
                    #if step % self.steps_per_train_log == 0:
                        gama = model.gama.eval()

                        print (np.shape(wb5))

                        db1 = np.array(wb1)[0:6, 0, :]
                        db2 = np.array(wb2)[0:6, 0, :]
                        db3 = np.array(wb3)[0:6, 0, :]
                        db4 = np.array(wb4)[0:6, 0, :]
                        db5 = np.array(wb5)[0:6, 0, :, 0]

                        print (np.round(db1, 5))
                        print (np.round(db2, 5))
                        print (np.round(db3, 5))
                        print (np.round(db4, 5))
                        print (np.round(db5, 5))
                        
                        grad = []
                        for g in grads[3:]:
                            v1 = np.sum(np.abs(g))
                            grad.append(v1)
                        print (grad[-7:])
                        #print (grad)

                        print ("epoch:%d, %d/%d, %.3f%%  %.3f" % (epoch, step, train_batch_num, float(step+1) /train_batch_num * 100, gama))

                        self.sample(batch['encoder_inputs'], batch['decoder_inputs'], batch['key_inputs'], outputs)
                        current_loss = total_loss / (step+1)
                        ppl = math.exp(current_loss) if current_loss < 300 else float('inf')
                        print("train loss: %.3f  ppl:%.2f  l2 loss: %.3f" % (current_loss, ppl, l2_loss))
                        time2 = time.time()
                        print ("%f secondes per iteration" % (float(time2-time1) /  self.steps_per_train_log))
                        time1 = time2

                        info = "train epoch: %d  %d/%d, %.3f%%,  loss: %.3f  ppl: %.2f " % ( epoch, step, train_batch_num, 
                            float(step+1) /train_batch_num * 100, current_loss, ppl )
                        fout = open("trainlog.txt", 'a')
                        fout.write(info + "\n")
                        fout.close()

                    '''
                    if (step+1) % gama_decay_steps == 0:
                        print ("gama decay!")
                        sess.run(model.gama_decay_op)
                    '''
                #
                current_epoch = int(model.global_step.eval() // train_batch_num)

                if epoch % self.epoch_per_valid_log == 0:
                    self.run_validation(sess, model, valid_batches, valid_batch_num, epoch)

                if epoch % self.epoch_per_checkpoint == 0:
                    # Save checkpoint and zero timer and loss.
                    print ("saving model...")
                    checkpoint_path = os.path.join("model", "poem.ckpt" + "_" + str(current_epoch))
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                
                print("shuffle data...")
                random.shuffle(train_batches)

def main(_):
    trainer = PoemTrainer()
    trainer.train()

if __name__ == "__main__":
    tf.app.run()
