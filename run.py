import os
import configargparse

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument('--src_domain',  type=str, required=True)
    parser.add_argument('--tgt_domain',  type=str, required=True)
    parser.add_argument('--method',  type=str, required=True)
    parser.add_argument('--seed',  type=str, required=True)
    parser.add_argument('--myweight',  type=float, required=True)
    parser.add_argument('--gpu',  type=int, default=0)
    return parser

parser = get_parser()
args = parser.parse_args()
src_domain=args.src_domain
tgt_domain=args.tgt_domain
method=args.method
seed=args.seed
myweight=args.myweight
gpu="CUDA_VISIBLE_DEVICES="+str(args.gpu)+" "

print("NEW TRAIN")
net_name=['DAN','DSAN','DANN','DAAN','DeepCoral','BNM','ONLY']
for i in range(len(net_name)):
    log_name=src_domain+'+'+tgt_domain+'+'+net_name[i]+'+'+method
    os.system('rm '+log_name+'.log')
    os.system(gpu+'nohup python3 main.py --config '+net_name[i]+'/'+net_name[i]+'.yaml --src_domain '+src_domain+' --tgt_domain '+tgt_domain+' --seed '+seed+' --tname transfer --method '+method+' --lr 3e-3 '+' --myweight  '+str(myweight)+' >>'+log_name+'.log 2>&1 &')

print(args)
