import os
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='EBR training')
    parser.add_argument('-method',  required=True, type=str,choices=['ebr','cebr'], help='dataset')
    parser.add_argument('-dataset',  required=True, type=str,choices=['iwdeen', 'wmtdeen', 'wmtende'], help='dataset')
    return parser  


# Loading arguments 
parser = get_parser()
args = parser.parse_args()



if args.dataset=='iwdeen':
   if args.method == 'ebr':
       os.system('bash downlaod_data-iwdeen.sh --ebr')
       os.system("echo 'ebr training for IWSLT DE-EN'")
       os.system('python train_ebr.py -method ebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_ebr_iwdeen')
    

   elif args.method == 'cebr':
       os.system('bash downlaod_data-iwdeen.sh')
       os.system("echo 'conditional-ebr training for IWSLT DE-EN'")
       os.system('python train_ebr.py -method cebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_cebr_iwdeen')
  


if args.dataset=='wmtdeen':
   if args.method == 'ebr':
      os.system('bash downlaod_data-wmtdeen.sh --ebr')
      os.system("echo 'ebr training for WMT DE-EN'")
      os.system('python train_ebr.py -method ebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_ebr_wmtdeen')
   elif args.method == 'cebr':
       os.system('bash downlaod_data-wmtdeen.sh')
       os.system("echo 'conditional-ebr training for WMT DE-EN'")
       os.system('python train_ebr.py -method cebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_cebr_wmtdeen')

if args.dataset=='wmtende':
   if args.method == 'ebr':
      os.system('bash downlaod_data-wmtende.sh --ebr')
      os.system("echo 'ebr training for WMT EN-DE'")
      os.system('python train_ebr.py -method ebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_ebr_wmtende')
   elif args.method == 'cebr':
       os.system('bash downlaod_data-wmtende.sh --ebr')
       os.system("echo 'conditional-ebr training for WMT EN-DE'")
       os.system('python train_ebr.py -method cebr -mixing_p 0.0 -path ./ -train train-ebr.csv -valid val-ebr.csv -test test-ebr.csv -save_dir checkpoints_cebr_wmtende')









   
   
 
   




