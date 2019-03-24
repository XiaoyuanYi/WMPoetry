from collections import namedtuple


HParams = namedtuple('HParams',
                     'vocab_size, emb_size, ph_emb_size, len_emb_size,'
                     'hidden_size, mem_size, his_mem_slots,' 
                     'global_trace_size, topic_trace_size, key_slots, sens_num, device, learning_rate,'
                     'bucket, max_gradient_norm, batch_size,'
                     'epoches_per_checkpoint, epoches_per_validate, steps_per_train_log,'
                     'sample_num, max_epoch, burn_down, decay_rate, l2_weight, keep_prob, annealing_epoches,'
                     'vocab_path, ivocab_path, train_data, valid_data, init_emb model_path, pre_model_path'
                     )

hps = HParams(
            vocab_size=-1, # It is to be replaced by true vocabulary size after loading dictionary.
            emb_size=256, hidden_size=512,
            ph_emb_size=64, # Phonology embedding size.
            len_emb_size=32, # Length embedding size.
            his_mem_slots=4, # Number of history memory slots.
            mem_size=512, # The size of each memory slot.
            global_trace_size=512,
            topic_trace_size=20,
            key_slots=4, # Number of topic memory slots.
            sens_num=4, # Number of lines in each poem.
            device='/gpu:0',
            learning_rate=0.001, max_gradient_norm=1.0,
            bucket=(9, 9), # Max length of a line.
            batch_size=32,
            epoches_per_checkpoint=1, epoches_per_validate=1,
            steps_per_train_log=200, max_epoch=10,
            burn_down=3, decay_rate=0.9,  l2_weight=1e-5, keep_prob=0.75,
            annealing_epoches=3, # Annealing of the sampled random bias
            sample_num=1, # Generate some poems during training for observation, with greedy search.
            vocab_path="data/train/vocab.pickle",
            ivocab_path="data/train/ivocab.pickle",
            train_data="data/train/train.pickle", # Training data path.
            valid_data="data/train/valid.pickle", # Validation data path.
            init_emb="", # The path of pre-trained word2vec embedding. Set it to '' if none.
            model_path="model/", # The path to save checkpoints.
            pre_model_path="premodel/" # The path to save checkpoints of the pre-trained model.
        )