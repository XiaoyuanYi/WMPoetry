from collections import namedtuple


HParams = namedtuple('HParams',
                     'vocab_size, emb_size, ph_emb_size, len_emb_size,'
                     'hidden_size, his_mem_size, his_mem_slots,' 
                     'global_trace_size, topic_trace_size, key_slots, sens_num, device, learning_rate,'
                     'bucket, max_gradient_norm, PAD_ID, batch_size, mode,'
                     'epoches_per_checkpoint, epoches_per_validate, steps_per_train_log,'
                     'sample_num, max_epoch, burn_down, decay_rate,'
                     'vocab_path, ivocab_path, train_data, valid_data, init_emb model_path'
                     )

hps = HParams(
            vocab_size=-1, # It is to be replaced by true vocabulary size after loading dictionary.
            emb_size=256, hidden_size=512,
            ph_emb_size=64, # Phonology embedding size.
            len_emb_size=32, # Length embedding size.
            his_mem_size=1024, # The size of a history memory slot, which must be 2*hidden_size.
            his_mem_slots=4, # Number of history memory slots.
            global_trace_size=512,
            topic_trace_size=20,
            key_slots=4, # Number of topic memory slots.
            sens_num=4, # Number of lines in each poem.
            device='/cpu:0',
            learning_rate=0.001, max_gradient_norm=1.0,
            bucket=(9, 9), # Max length of a line.
            PAD_ID=-1, # The idx of PAD symbol, which is to be replaced by true idx after loading dictionary.
            batch_size=16,
            mode='train', # train or decode
            epoches_per_checkpoint=1, epoches_per_validate=1,
            steps_per_train_log=5, max_epoch=8,
            burn_down=2, decay_rate=0.9,
            sample_num=1, # Generate some poems during training for observation, with greedy search.
            vocab_path="data/train/vocab.pkl",
            ivocab_path="data/train/ivocab.pkl",
            train_data="data/train/train.pkl", # Training data path.
            valid_data="data/train/valid.pkl", # Validation data path.
            init_emb="data/train/init_emb.npy", # The path of pre-trained word2vec embedding. Set it to '' if none.
            model_path="model/" # The path to save checkpoints.

        )