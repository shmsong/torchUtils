r"""
Running this script:

    run with clearing cache by big-tensor:
    
           nvprof python big_tensor.py  --executor profiling --sizes 1024 8192 --big_tensor 2>&1 | egrep '(CudaCodeGen|iter)'
           
    
    run without clearning cache by big-tensor:
    
           nvprof python big_tensor.py  --executor profiling --sizes 1024 8192 2>&1 | egrep '(CudaCodeGen|iter)'
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

os.environ['PYTORCH_CUDA_FUSER_DISABLE_FALLBACK'] = '1'
os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '0'
os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '4'
#os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '1'
#os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '0'

import argparse
import torch
import itertools
import numpy as np
import time

def initialize(args):
  assert args.fuser in ['nv', 'te', 'old', 'jit', 'eager']
  if args.fuser == 'te':
      torch._C._jit_set_profiling_executor(True)
      torch._C._jit_set_profiling_mode(True)
      torch._C._jit_set_bailout_depth(20)
      torch._C._jit_override_can_fuse_on_cpu(False)
      torch._C._jit_override_can_fuse_on_gpu(False)
      torch._C._jit_set_texpr_fuser_enabled(True)
      args.fuserNode = 'tensorexpr::Group'
  elif args.fuser == 'old':
      torch._C._jit_set_profiling_executor(False)
      torch._C._jit_set_profiling_mode(False)
      torch._C._jit_override_can_fuse_on_gpu(True)
      torch._C._jit_set_texpr_fuser_enabled(False)
      args.fuserNode = 'prim::FusionGroup'
  elif args.fuser == 'jit':
      torch._C._jit_set_profiling_executor(False)
      torch._C._jit_set_profiling_mode(False)
      torch._C._jit_override_can_fuse_on_gpu(False)
      torch._C._jit_override_can_fuse_on_cpu(False)
      torch._C._jit_set_texpr_fuser_enabled(False)
      args.fuserNode = None
  elif args.fuser == 'nv':
      torch._C._jit_override_can_fuse_on_cpu(False)
      torch._C._jit_override_can_fuse_on_gpu(False)
      torch._C._jit_set_profiling_executor(True)
      torch._C._jit_set_profiling_mode(True)
      torch._C._jit_set_bailout_depth(20)
      torch._C._jit_set_nvfuser_enabled(True)
      args.fuserNode = 'prim::CudaFusionGroup'
  elif args.fuser == 'eager':
      args.fuserNode = None

  # --executor overrides settings of --fuser
  assert args.executor in ['profiling', 'simple', 'legacy']
  if args.executor == 'profiling':
      torch._C._jit_set_profiling_executor(True)
      torch._C._jit_set_profiling_mode(True)
      torch._C._jit_set_bailout_depth(20)
  elif args.executor == 'simple':
      torch._C._jit_set_profiling_executor(True)
      torch._C._jit_set_profiling_mode(False)
  elif args.executor == 'legacy':
      torch._C._jit_set_profiling_executor(False)
      torch._C._jit_set_profiling_mode(False)


def run(args,big_tensor=True):
    def model(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        o = torch.mul(x, y)
        o = torch.mul(o, z)
        o = torch.relu(o)
        return o

    model_script = torch.jit.script(model)

    if big_tensor:
        _big_tensor = torch.randn(1024*16, dtype=torch.float32, device="cuda")
    
    x = torch.randn(args.sizes, dtype=torch.float32, device="cuda")
    y = torch.randn(args.sizes, dtype=torch.float32, device="cuda")
    z = torch.randn(args.sizes, dtype=torch.float32, device="cuda")
    inputs = [x,y,z]

    elapsed_time = 0

    with torch.no_grad():
        for i in range(args.warmup):
            jit_outs = model_script(*inputs)

        for i in range(args.nloops):
            torch.cuda.synchronize()
            start = time.time()
            jit_outs = model_script(*inputs)
            torch.cuda.synchronize()
            end = time.time()
            elapsed_time += end - start
            # clear cache
            if big_tensor:
                _big_tensor = _big_tensor * 1.0

    iter_elapsed_time = elapsed_time / args.nloops

    # check output:
    eager_outs = model(*inputs)
    for eager_out, jit_out in zip(eager_outs, jit_outs):
        assert(eager_out.dtype == jit_out.dtype)
        error = 1e-5
        assert(torch.allclose(eager_out,jit_out,error,error))
  
    return iter_elapsed_time

def main(args):
    # set params
    initialize(args)

    # run experiment
    iter_time = run(args,big_tensor=args.big_tensor)

    # format output
    w = 'w/' if args.big_tensor else 'w/o'
    print('average iter time {} big tensor: {:.2e} us'.format(w,iter_time*1e6))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Profile RNNs')

  parser.add_argument('--warmup', default='100', type=int)
  parser.add_argument('--nloops', default='500', type=int)
  parser.add_argument('--np_seed', default='128', type=int)
  parser.add_argument('--big_tensor',  action='store_true')

  # pw specific
  parser.add_argument('--sizes', nargs='+', default=[100, 200], type=int)

  parser.add_argument('--executor', default=None, type=str,
                      help='The executor to use. One of: legacy, simple, profiling')
  parser.add_argument('--fuser', default='nv', type=str,
                      help='The fuser backend to use. One of: te, old, nv or jit, eager')
  args = parser.parse_args()

  main(args)
