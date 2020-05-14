import tensorflow as tf
import numpy as np
import os 
import agent
import config as cfg 

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.Session(config=config)
filewriter=tf.summary.FileWriter(cfg.params['log_dir']+cfg.params['level_name'])
PG_worker=agent.Vanilla_Agent(session,filewriter)

with session:
    session.run(tf.global_variables_initializer())
    PG_worker.work()
