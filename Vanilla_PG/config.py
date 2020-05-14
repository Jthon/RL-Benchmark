import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument("--level_name",type=str,default="LunarLander-v2")
parser.add_argument("--agent",type=str,default="random")
parser.add_argument("--seed",type=int,default=2330)

parser.add_argument("--model_dir",type=str,default="./model/")
parser.add_argument("--log_dir",type=str,default="./log/")
parser.add_argument("--result_dir",type=str,default="./result/")

parser.add_argument("--episode",type=int,default=20000)
parser.add_argument("--save_episode",type=int,default=100)
parser.add_argument("--lr_rate",type=float,default=1e-4)
parser.add_argument("--gamma",type=float,default=0.99)
args = parser.parse_args()
# convert args to dictionary
params = vars(args)

if not os.path.exists(params['model_dir']):
    os.mkdir(params['model_dir'])
if not os.path.exists(params['model_dir']+params['level_name']):
    os.mkdir(params['model_dir']+params['level_name'])

if not os.path.exists(params['result_dir']):
    os.mkdir(params['result_dir'])
if not os.path.exists(params['result_dir']+params['level_name']):
    os.mkdir(params['result_dir']+params['level_name'])

if not os.path.exists(params['log_dir']):
    os.mkdir(params['log_dir'])
