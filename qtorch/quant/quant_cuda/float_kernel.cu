#include "quant_kernel.h"
#include "bit_helper.cu"
#include <cstdio>
#include <cuda_fp16.h>

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int quantize = round_bitwise_stochastic(old_num, rand_prob, man_bits);
    quantize = clip_exponent(exp_bits, man_bits, old_num, quantize);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int exp = (old_num & 0x7F800000)>>23;
    unsigned int man = (old_num & 0x007FFFFF);
    int true_exp = (int)exp - 127;
    if(exp > 0) { // normal float 
       man = man | (1<<23); 
       const int DIY_bias = (1<<(exp_bits-1)) - 1;
       int new_e = true_exp + DIY_bias;
       if(new_e > 0) { // normal number for DIY precision
          // round man
          if((man & 1<<(23 - man_bits - 1)) == 0) // just round to lower
          {
             man = man & ~((1<<(23 - man_bits)) - 1);
          }
          else
          {
             if((man & ((1<<(23 - man_bits - 1)) -1)) != 0) { // just round to upper
                man = man + (1<<(23 - man_bits - 1));
                // check if the high position is changed
                if((man & (1<<(23+1))) == 0) // have not changed
                {
                    man = man & (~((1<<(23 - man_bits)) - 1));
                }
                else {
                    man = man >> 1;
                    man = man & ~((1<<(23 - man_bits)) - 1);
                    new_e += 1;
                }
             }
             else { // round to nearest even
                if(man & (1<<(23 - man_bits)) == 0) // just truncation
                    man = man & ~((1<<(23 - man_bits)) - 1);
                else {
                    man = man + (1<<(23 - man_bits - 1));
                    // check if the high position is changed
                    if(man & (1<<(23+1)) == 0) // have not changed
                    {
                        man = man & ~((1<<(23 - man_bits)) - 1);
                    }
                    else {
                        man = man >> 1;
                        man = man & ~((1<<(23 - man_bits)) - 1);
                        new_e += 1;
                    }
                }
             }
          }
          new_e -= DIY_bias;
       }
       else { //subnormal number for DIY precision
          man = man>>(-new_e);
          new_e = - DIY_bias;
          //TODO: add round part
       }
       // TODO: maybe e will be 0xFFF, this will cause bug
       if(new_e >=0)
            o[index] = ((float)man)/(1<<23) * (1<<new_e);
       else
            o[index] = ((float)man)/(1<<23) / (1<<(-new_e));
       if(a[index] < 0)
          o[index] = -o[index];
    }
  }
}
