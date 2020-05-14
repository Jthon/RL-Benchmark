import tensorflow as tf
import gym_env as env
import model
import config as cfg
import numpy as np
import imageio
class Vanilla_Agent:
    def __init__(self,session,logger):
        self.env=env.gym_env()

        self.ob_dim=self.env.observation_dim
        self.ac_dim=self.env.action_dim
        self.discrete=self.env.discrete

        self.model=model.Vanilla_PG(self.ob_dim,self.ac_dim,self.discrete)
        self.score_var,self.score_log=self.logVariable()
        self.buffer=[]
        self.current_episode=0

        self.session=session
        self.logger=logger
        self.saver=tf.train.Saver(max_to_keep=1)
        
        self.avg=0
    def discount_return(self,reward_buffer,gamma):
        result=np.zeros(shape=(reward_buffer.shape[0]+1),dtype=np.float32)
        for i in range(1,reward_buffer.shape[0]+1):
            result[reward_buffer.shape[0]-i]=gamma*result[reward_buffer.shape[0]-i+1]+reward_buffer[reward_buffer.shape[0]-i]
        return result[0:reward_buffer.shape[0]]
    def update(self):
        obs_buffer=[]
        act_buffer=[]
        rew_buffer=[]
        for element in self.buffer:
            obs_buffer.append(element[0])
            act_buffer.append(element[1])
            rew_buffer.append(element[3])
        obs_buffer=np.array(obs_buffer)
        act_buffer=np.array(act_buffer)
        rew_buffer=np.array(rew_buffer)
        discount_return=self.discount_return(rew_buffer,cfg.params['gamma'])
        baseline=self.session.run(self.model.baseline,feed_dict={self.model.observation:obs_buffer})[:,0]
        self.session.run([self.model.p_train_op,self.model.b_train_op],feed_dict={self.model.observation:obs_buffer,self.model.action:act_buffer,self.model.a_input:discount_return-baseline,self.model.v_input:discount_return})
    def work(self):
        while self.current_episode<cfg.params["episode"]:
            image_list=[]
            self.buffer=[]
            score=0
            done=False
            obs=self.env.reset()
            while not done:
                image_list.append(self.env.render())
                action=self.action(obs)
                next_obs,reward,done=self.env.step(action)
                self.buffer.append((obs,action,next_obs,reward,done))
                obs=next_obs
                score+=reward
            self.current_episode+=1
            self.avg=self.avg+(score-self.avg)/self.current_episode
            print("Episode:%d get score %d"%(self.current_episode,score))
            summary=self.session.run(self.score_log,feed_dict={self.score_var:score})
            self.logger.add_summary(summary,self.current_episode)
            self.update()
            
            if self.current_episode%cfg.params["save_episode"]==0:
                self.saver.save(self.session,cfg.params['model_dir']+cfg.params['level_name']+"/epi="+str(self.current_episode)+".ckpt")
                self.makegif(image_list,cfg.params["result_dir"]+cfg.params["level_name"]+"/epi="+str(self.current_episode)+".gif",0.02)

    def makegif(self,image_list,gif_path,duration):
        imageio.mimsave(gif_path, image_list, 'GIF', duration=duration)

    def logVariable(self):
        score_var=tf.Variable(0.0)
        score_log=tf.summary.scalar("score",score_var)
        return score_var,score_log
    
    def action(self,observation):
        if self.discrete==True:
            prob=self.session.run(self.model.policy,feed_dict={self.model.observation:[observation]})[0]
            act=np.random.choice(self.ac_dim,p=prob)
        return act