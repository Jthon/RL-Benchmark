import tensorflow as tf
import tensorflow.layers as layers
import config as cfg
class Vanilla_PG:
    def __init__(self,ob_dim,ac_dim,discrete):
        self.observation=tf.placeholder(dtype=tf.float32,shape=[None,ob_dim])
        self.a_input=tf.placeholder(dtype=tf.float32,shape=[None])
        self.v_input=tf.placeholder(dtype=tf.float32,shape=[None])
        if discrete==True:
            self.action=tf.placeholder(dtype=tf.int64,shape=[None])
            self.action_onehot=tf.one_hot(self.action,ac_dim)
        else:
            self.action=tf.placeholder(dtype=tf.float32,shape=[None,ac_dim])
        self.ob_dim=ob_dim
        self.ac_dim=ac_dim
        self.discrete=discrete
        self.kernel_init=tf.truncated_normal_initializer(stddev=0.02)
        self.bias_init=tf.constant_initializer(value=0)
        self.BuildNet()
    def MLP_Policy(self,observation):
        with tf.variable_scope("MLP_Policy"):
            fc1=layers.dense(observation,128,activation=tf.nn.leaky_relu,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
            fc2=layers.dense(fc1,128,activation=tf.nn.leaky_relu,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
            if self.discrete==True:
                self.policy=layers.dense(fc2,self.ac_dim,activation=tf.nn.softmax,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
            else:
                self.mean=layers.dense(fc2,self.ac_dim,activation=None,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
                self.policy=tf.random_normal(shape=(self.observation.shape[0],self.ac_dim),mean=self.mean,stddev=tf.ones_like(self.mean))
    def MLP_Baseline(self,observation):
        with tf.variable_scope("Baseline_estimate"):
            fc1=layers.dense(observation,128,activation=tf.nn.leaky_relu,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
            fc2=layers.dense(fc1,128,activation=tf.nn.leaky_relu,kernel_initializer=self.kernel_init,bias_initializer=self.bias_init)
            self.baseline=layers.dense(fc2,1)
    def BuildNet(self):
        self.MLP_Policy(self.observation)
        self.MLP_Baseline(self.observation)
        if self.discrete==True:
            self.choice_prob=tf.reduce_sum(self.policy*self.action_onehot,axis=1)
            self.log_prob=tf.log(self.choice_prob+1e-7)
            self.policy_loss=tf.reduce_sum(-self.a_input*self.log_prob)
            self.baseline_loss=tf.reduce_mean(tf.square(self.baseline-tf.reshape(self.v_input,(-1,1))))
            self.p_train_op=tf.train.AdamOptimizer(cfg.params['lr_rate']).minimize(self.policy_loss)
            self.b_train_op=tf.train.AdamOptimizer(cfg.params["lr_rate"]).minimize(self.baseline_loss)


