from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time

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
