
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
    parser.add_argument('--name',  type=str, required=True)
    parser.add_argument('--method',  type=str, required=True)
    parser.add_argument('--seed',  type=str, required=True)
    parser.add_argument('--myweight',  type=float, required=True)
    parser.add_argument('--gpu',  type=int, default=0)
    return parser

parser = get_parser()
args = parser.parse_args()
net_name=args.name
method=args.method
seed=args.seed
myweight=args.myweight
gpu="CUDA_VISIBLE_DEVICES="+str(args.gpu)+" "

print("NEW TRAIN")
src_domain=["UCI","UCI","WISDM","WISDM","PAMAP2","PAMAP2","77G","77G","45","45"]
tgt_domain=["WISDM","PAMAP2","UCI","PAMAP2","UCI","WISDM","own","63G","own","63G"]
for i in range(6):
    print(i)
    log_name=src_domain[i]+'+'+tgt_domain[i]+'+'+net_name+'+'+method
    os.system('rm '+log_name+'.log')
    os.system(gpu+'nohup python3 main_dwl.py --config '+net_name+'/'+net_name+'.yaml  --src_domain '+src_domain[i]+' --tgt_domain '+tgt_domain[i]+' --seed '+seed+' --tname transfer --method '+method+' --lr 3e-3 '+' --myweight  '+str(myweight)+' >>'+log_name+'.log 2>&1 &')

print(args)
