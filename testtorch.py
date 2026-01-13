import torch
import os
os.environ['CUDA_HOME']='e:/PC/cuda126'
print(torch.cuda.is_available())