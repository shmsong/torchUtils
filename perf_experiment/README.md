**How to Run**

Every parameter is in run.sh. Edit and source run.sh

**512 x 1024 Pointwise graph:**

*With big tensor*

  End to end:
  
              average iter time w/ big tensor: 2.45e+02 us
              
  Kernel time:
  
              GPU activities:   42.25%  9.6536ms       599  16.116us  15.777us  17.120us  CudaCodeGen::kernel1(CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>)

*Without big tensor*

  End to end:
  
              average iter time w/o big tensor: 2.38e+02 us
              
  Kernel time:
  
              GPU activities:   43.97%  9.6258ms       599  16.069us  15.648us  16.896us  CudaCodeGen::kernel1(CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>)

**1024 x 1024 Pointwise graph:**

*With big tensor*

  End to end:
  
              average iter time w/ big tensor: 2.57e+02 us
              
  Kernel time:
  
              GPU activities:   41.37%  17.946ms       599  29.959us  29.440us  31.232us  CudaCodeGen::kernel1(CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>)

*Without big tensor*

  End to end:
  
              average iter time w/o big tensor: 2.50e+02 us
              
  Kernel time:
  
              GPU activities:   42.20%  17.948ms       599  29.964us  29.472us  31.008us  CudaCodeGen::kernel1(CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>, CudaCodeGen::Tensor<float, int=2>)
