import warnings
import pprint

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
from parser import  test_parser,train_parser
import pretty_errors
from Solver import CISolver, BaselineSolver

args = train_parser()

if __name__ == '__main__':
    print(args)
    if args.net.lower() in ['macr', 'pda', 'tide', 'dice']:
        solver = BaselineSolver(args)
    else:
        solver = CISolver(args)

    if not args.test:
        solver.train()
    else:
        solver.test()
    
 

