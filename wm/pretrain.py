from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time

import numpy as np
import tensorflow as tf

import graphs
from tool import PoetryTool
from config import hps

class PoemTrainer(object):

    def __init__(self):
        # Construct hyper-parameter
        self.hps = hps

        self.enc_len = hps.bucket[0]
        self.dec_len = hps.bucket[1]
        self.sens_num = hps.sens_num
        self.key_slots = hps.key_slots

        self.tool = PoetryTool(sens_num=hps.sens_num,
            key_slots=hps.key_slots, enc_len=hps.bucket[0],
            dec_len=hps.bucket[1])
        # If there isn't a pre-trained word embedding, just
        #   set it to None, then the word embedding
        #   will be initialized with a norm distribution.
        if hps.init_emb == '':
            self.init_emb = None
        else:
            self.init_emb = np.load(self.hps.init_emb)
            print ("init_emb_size: %s" % str(np.shape(self.init_emb)))
        self.tool.load_dic(hps.vocab_path, hps.ivocab_path)

        vocab_size = self.tool.get_vocab_size()
        assert vocab_size > 0
        self.PAD_ID = self.tool.get_PAD_ID()
        assert self.PAD_ID == 0

        self.hps = self.hps._replace(vocab_size=vocab_size)

        print("Params  sets: ")
        print (self.hps)
        print("___________________")


    def create_model(self, sess, path):
        ckpt = tf.train.get_checkpoint_state(path)
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)
        #print (ckpt.model_checkpoint_path)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

    def step(self, sess, data_dic, model, valid):
        '''For training one batch'''
        keep_prob = 1.0 if valid else self.hps.keep_prob

        input_feed = {}
        input_feed[model.keep_prob] = keep_prob

        if len(data_dic['enc_inps']) != self.enc_len:
            raise ValueError("Encoder length must be equal %d != %d." % 
                (len(data_dic['enc_inps']), self.enc_len))

        if len(data_dic['dec_inps']) != self.dec_len:
            raise ValueError("Decoder length must be equal %d != %d. " %
                (len(data_dic['dec_inps']), self.dec_len))
            
        if len(data_dic['trg_weights']) != self.dec_len:
            raise ValueError("Weights length must be equal %d != %d." %
                (len(data_dic['trg_weights']), self.dec_len))
        
        for l in range(self.enc_len):
            input_feed[model.enc_inps[0][l].name] = data_dic['enc_inps'][l]

        for l in range(self.dec_len):
            input_feed[model.dec_inps[0][l].name] = data_dic['dec_inps'][l]
            input_feed[model.trg_weights[0][l].name] = data_dic['trg_weights'][l]
            input_feed[model.len_inps[0][l].name] = data_dic['len_inps'][l]
            input_feed[model.ph_inps[0][l].name] = data_dic['ph_inps'][l]

        last_target = model.dec_inps[0][self.dec_len].name
        input_feed[last_target] = np.ones([self.hps.batch_size], dtype=np.int32) * self.PAD_ID

        output_feed = []       
        for l in range(self.dec_len):  # Output logits.
            output_feed.append(self.outs_op[l])

        output_feed += [self.gen_loss_op, self.l2_loss_op, self.global_step_op]

        if not valid:
            output_feed += [self.train_op] 

        outputs = sess.run(output_feed, input_feed)

        n = self.dec_len
        logits = outputs[:n]
        return logits, outputs[n], outputs[n+1], outputs[n+2]

    def sample(self, enc_inps, dec_inps, outputs):

        sample_num = min(self.hps.sample_num, self.hps.batch_size)

        # Random select some examples
        idxes = random.sample(list(range(0, self.hps.batch_size)), 
            sample_num)

        #
        for idx in idxes:
            # Build lines
            inputs = [c[idx] for c in enc_inps]
            sline = "".join(self.tool.idxes2chars(inputs))

            target = [c[idx] for c in dec_inps]
            tline = "".join(self.tool.idxes2chars(target))

            outline = [c[idx] for c in outputs]
            outline = self.tool.greedy_search(outline)

            print(sline.ljust(30) + " # " + tline.ljust(30) + " # " + outline.ljust(30) + " # ")


    def run_validation(self, sess, model, valid_batches, valid_batch_num, epoch):
        print("run validation...")
        total_gen_loss = 0.0
        total_l2_loss = 0.0
        for step in range(0, valid_batch_num):
            batch = valid_batches[step]
            _, gen_loss, l2_loss, _, = self.step(sess, batch, model, True)
            total_gen_loss += gen_loss
            total_l2_loss += l2_loss
        total_gen_loss /= valid_batch_num
        total_l2_loss /= valid_batch_num
        info = "validation epoch: %d  loss: %.3f  ppl: %.2f, l2 loss: %.3f" % \
            (epoch, total_gen_loss, np.exp(total_gen_loss), total_l2_loss)
        print (info)
        fout = open("validlog.txt", 'a')
        fout.write(info + "\n")
        fout.close()

    def train(self):
        print ("building data...")
        train_batches, train_batch_num = self.tool.build_data(self.hps.train_data,
            self.hps.batch_size, 'pre_train')
        valid_batches, valid_batch_num = self.tool.build_data(self.hps.valid_data,
            self.hps.batch_size, 'pre_train')

        print ("train batch num: %d" % (train_batch_num))
        print ("valid batch num: %d" % (valid_batch_num))

        input("Please check the parameters and press enter to continue>")

        model = graphs.WorkingMemoryModel(self.hps)
        self.train_op, self.outs_op, self.gen_loss_op, self.l2_loss_op, \
            self.global_step_op, = model.pre_training()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

            self.create_model(sess, self.hps.pre_model_path)

            # check the initialization
            print ("uninitialized_variables")
            uni = sess.run(tf.report_uninitialized_variables(tf.global_variables(),
                name='report_uninitialized_variables'))
            print (len(uni))

            burn_down = min(self.hps.burn_down, self.hps.max_epoch)
            total_gen_loss = 0.0
            total_l2_loss = 0.0
            total_steps = 0
            time1 = time.time()

            for epoch in range(1, self.hps.max_epoch+1):

                for step in range(0, train_batch_num):
                    batch = train_batches[step]

                    total_steps += 1

                    logits, gen_loss, l2_loss, global_step = self.step(sess, batch, model, False)

                    total_gen_loss += gen_loss
                    total_l2_loss += l2_loss

                    if step % self.hps.steps_per_train_log == 0:
                        time2 = time.time()

                        time_cost = float(time2-time1) / self.hps.steps_per_train_log

                        cur_gen_loss = total_gen_loss / (total_steps+1)
                        cur_ppl = math.exp(cur_gen_loss)
                        cur_l2_loss = total_l2_loss / (total_steps+1)

                        self.sample(batch['enc_inps'], batch['dec_inps'], logits)

                        process_info = "epoch: %d, %d/%d %.3f%%, %.3f s per iter" % (epoch, step, train_batch_num,
                            float(step+1) /train_batch_num * 100, time_cost)
                        
                        train_info = "train loss: %.3f  ppl:%.2f, l2 loss: %.3f, lr:%.4f." \
                            % (cur_gen_loss, cur_ppl, cur_l2_loss, model.learning_rate.eval())
                        print (process_info)
                        print(train_info)
                        print("______________________")
                        
                        info = process_info + " " + train_info
                        fout = open("trainlog.txt", 'a')
                        fout.write(info + "\n")
                        fout.close()

                        time1 = time.time()


                current_epoch = int(global_step // train_batch_num)
                
                lr0 = model.learning_rate.eval()
                if epoch > burn_down:
                    print ("lr decay...")
                    sess.run(model.learning_rate_decay_op)
                lr1 = model.learning_rate.eval()
                print ("%.4f to %.4f" % (lr0, lr1))

                if epoch % self.hps.epoches_per_validate == 0:
                    self.run_validation(sess, model, valid_batches, valid_batch_num, current_epoch)

                if epoch % self.hps.epoches_per_checkpoint == 0:
                    # Save checkpoint
                    print("saving model...")
                    checkpoint_path = os.path.join(self.hps.pre_model_path, "poem.ckpt" + "_" + str(current_epoch))
                    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1 )
                    saver.save(sess, checkpoint_path, global_step=global_step)
                
                print("shuffle data...")
                random.shuffle(train_batches)

def main(_):
    trainer = PoemTrainer()
    trainer.train()

if __name__ == "__main__":
    tf.app.run()
