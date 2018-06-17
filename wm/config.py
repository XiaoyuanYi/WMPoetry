from collections import namedtuple


HParams = namedtuple('HParams',
                     'vocab_size, emb_size, ph_emb_size, len_emb_size'
                     'hidden_size, his_mem_size, his_mem_slots,' 
                     'global_trace_size, topic_trace_size, key_slots, sens_num, device, learning_rate,'
                     'bucket, max_gradient_norm, buckets, PAD_ID, batch_size, mode,'
                     'epoches_per_checkpoint, epoches_per_validate, steps_per_train_log,'
                     'sample_num, max_epoch, burn_down, decay_rate'
                     )

hps = HParams(
            vocab_size=10000, # It is to be replaced by true vocabulary size after loading dictionary
            emb_size=256, hidden_size=512,
            ph_emb_size=64, # Phonology embedding size
            len_emb_size=32, # Length embedding size
            his_mem_size=1024, # Must be 2*hidden_size
            his_mem_slots=4, # Number of history memory slot
            global_trace_size=512,
            topic_trace_size=20,
            key_slots=4, # Number of topic memory slot
            sens_num=4, # Number of lines in each poem
            device='/cpu:0',
            learning_rate=0.001, max_gradient_norm=1.0,
            bucket=(9, 9), # Max lengtn of a line
            PAD_ID=0, # idx of PAD symbol
            batch_size=8,
            mode='train', # train or decode
            epoches_per_checkpoint=1, epoches_per_validate=1,
            steps_per_train_log=200, max_epoch=8,
            burn_down=4, decay_rate=0.9,
            sample_num=1
        )