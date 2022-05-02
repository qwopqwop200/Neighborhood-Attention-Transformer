#This is NAT implemented using cupy.
nh_attn_forward_q_k = '''
extern "C"
__global__ void nh_attn_forward_q_k(const ${Dtype}* query, const ${Dtype}* key, const ${Dtype}* bias, ${Dtype}* attn, \
                                    const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${window_sq_size};
        const int k = index - indtmp1 * ${window_sq_size};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 *height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        const int kh = k / ${window_size};
        const int kw = k - kh * ${window_size};

        int ph = ${neighborhood_size};
        int pw = ${neighborhood_size};
        int nh = h - ${neighborhood_size};
        int nw = w - ${neighborhood_size};

        if (nh < 0){
            nh = 0;
            ph = ${window_size} - 1 - h;
        }
        else if (h + ${neighborhood_size} >= height){
            nh = height - ${window_size};
            ph = height - h - 1;
        }

        if (nw < 0){
            nw = 0;
            pw = ${window_size} - 1 - w;
        }
        else if (w + ${neighborhood_size} >= width){
            nw = width - ${window_size};
            pw = width - w - 1;
        }

        const int batch_idx = b * ${num_heads} + n_h;
        const int q_idx = ((batch_idx * height + h) * width + w) * ${channels};
        const int k_idx = ((batch_idx * height + (kh+nh)) * width + (kw+nw)) * ${channels};
        const int b_idx = (n_h * ${bias_size} + (ph+kh)) * ${bias_size} + (pw+kw);

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int c=0; c < ${channels}; ++c){
            update_value += query[q_idx+c] * key[k_idx+c];
        }
        update_value += bias[b_idx];
        attn[index] = update_value;
    }
}
'''

nh_attn_backward_query = '''
extern "C"
__global__ void nh_attn_backward_query(const ${Dtype}* const key, const ${Dtype}* const d_attn, ${Dtype}* const d_query, \
                                       const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${channels};
        const int c = index - indtmp1 * ${channels};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        int nh = max(h - ${neighborhood_size}, 0) + (h + ${neighborhood_size} >= height) * (height - h - ${neighborhood_size} - 1);
        int nw = max(w - ${neighborhood_size}, 0) + (w + ${neighborhood_size} >= width) * (width - w - ${neighborhood_size} - 1);

        const int batch_idx = b * ${num_heads} + n_h;
        int a_idx = ((batch_idx * height + h) * width + w)*${window_sq_size};

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int xh=nh; xh < nh + ${window_size}; ++xh){
            #pragma unroll
            for (int xw=nw; xw < nw + ${window_size}; ++xw){
                const int k_idx = (((batch_idx * height + xh) * width + xw) * ${channels} + c);
                update_value += d_attn[a_idx] * key[k_idx];
                ++a_idx;
            }
        }
        d_query[index] = update_value;
    }
}
'''

nh_attn_backward_key = '''
extern "C"
__global__ void nh_attn_backward_key(const ${Dtype}* const query, const ${Dtype}* const d_attn, ${Dtype}* const d_key, \ 
                                     const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${channels};
        const int c = index - indtmp1 * ${channels};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        int nh = max(h - ${neighborhood_size}, 0) + (h + ${neighborhood_size} >= height) * (height - h - ${neighborhood_size} - 1);
        int nw = max(w - ${neighborhood_size}, 0) + (w + ${neighborhood_size} >= width) * (width - w - ${neighborhood_size} - 1);
        
        const int batch_idx = b * ${num_heads} + n_h;
        int a_idx = ((batch_idx * height + h) * width + w)*${window_sq_size};

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int xh=nh; xh < nh + ${window_size}; ++xh){
            #pragma unroll
            for (int xw=nw; xw < nw + ${window_size}; ++xw){
                const int k_idx = (((batch_idx * height + xh) * width + xw) * ${channels} + c);
                atomicAdd(&d_key[k_idx], query[index] * d_attn[a_idx]);
                ++a_idx;
            }
        }
    }
}
'''

nh_attn_backward_bias = '''
extern "C"
__global__ void nh_attn_backward_bias(const ${Dtype}* const d_attn, ${Dtype}* const d_bias, \
                                      const int nthreads, const int height, const int width) {
      CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${window_sq_size};
        const int k = index - indtmp1 * ${window_sq_size};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 *height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};

        const int kh = k / ${window_size};
        const int kw = k - kh * ${window_size};

        int ph = ${neighborhood_size};
        int pw = ${neighborhood_size};

        if (h < ${neighborhood_size}){
            ph = ${window_size} - 1 - h;
        }
        else if (h + ${neighborhood_size} >= height){
            ph = height - h - 1;
        }

        if (w < ${neighborhood_size}){
            pw = window_size - 1 - w;
        }
        else if (w + ${neighborhood_size} >= width){
            pw = width - w - 1;
        }

        const int b_idx = (n_h * ${bias_size} + (ph+kh)) * ${bias_size} + (pw+kw);

        atomicAdd(&d_bias[b_idx], d_attn[index]);
    }
}
'''
nh_attn_forward_attn_v = '''
extern "C"
__global__ void nh_attn_forward_attn_v(const ${Dtype}* attn, const ${Dtype}* value, ${Dtype}* out, \
                                       const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${channels};
        const int c = index - indtmp1 * ${channels};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        int nh = max(h - ${neighborhood_size}, 0) + (h + ${neighborhood_size} >= height) * (height - h - ${neighborhood_size} - 1);
        int nw = max(w - ${neighborhood_size}, 0) + (w + ${neighborhood_size} >= width) * (width - w - ${neighborhood_size} - 1);

        const int batch_idx = b * ${num_heads} + n_h;
        int a_idx = ((batch_idx * height + h) * width + w)*${window_sq_size};

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int xh=nh; xh < nh + ${window_size}; ++xh){
            #pragma unroll
            for (int xw=nw; xw < nw + ${window_size}; ++xw){
                const int v_idx = (((batch_idx * height + xh) * width + xw) * ${channels} + c);
                update_value += attn[a_idx] * value[v_idx];
                ++a_idx;
            }
        }
        out[index] = update_value;
    }
}
'''
nh_attn_backward_attn = '''
extern "C"
__global__ void nh_attn_backward_attn(const ${Dtype}* const value, const ${Dtype}* const d_out, ${Dtype}* const d_attn, \
                                      const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${window_sq_size};
        const int k = index - indtmp1 * ${window_sq_size};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 *height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        const int kh = k / ${window_size};
        const int kw = k - kh * ${window_size};

        int nh = max(h - ${neighborhood_size}, 0) + (h + ${neighborhood_size} >= height) * (height - h - ${neighborhood_size} - 1);
        int nw = max(w - ${neighborhood_size}, 0) + (w + ${neighborhood_size} >= width) * (width - w - ${neighborhood_size} - 1);
        
        const int batch_idx = b * ${num_heads} + n_h;
        const int o_idx = ((batch_idx * height + h) * width + w) * ${channels};
        const int v_idx = ((batch_idx * height + (nh+kh))* width + (nw+kw)) * ${channels};

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int c=0; c < ${channels}; ++c){
            update_value += d_out[o_idx+c] * value[v_idx+c];
        }
        d_attn[index] = update_value;
    }
}
'''

nh_attn_backward_value = '''
extern "C"
__global__ void nh_attn_backward_value(const ${Dtype}* const attn, const ${Dtype}* const d_out, ${Dtype}* const d_value \
                                       const int nthreads, const int height, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int indtmp1 = index/${channels};
        const int c = index - indtmp1 * ${channels};
        int indtmp2 = indtmp1/width;
        const int w = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int h = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/${num_heads};
        const int n_h = indtmp1 - indtmp2 * ${num_heads};
        const int b = indtmp2;

        int nh = max(h - ${neighborhood_size}, 0) + (h + ${neighborhood_size} >= height) * (height - h - ${neighborhood_size} - 1);
        int nw = max(w - ${neighborhood_size}, 0) + (w + ${neighborhood_size} >= width) * (width - w - ${neighborhood_size} - 1);
        
        const int batch_idx = b * num_heads + n_h;
        int a_idx = ((batch_idx * height + h) * width + w)*${window_sq_size};

        ${Dtype} update_value = 0;
        #pragma unroll
        for (int xh=nh; xh < nh + ${window_size}; ++xh){
            #pragma unroll
            for (int xw=nw; xw < nw + ${window_size}; ++xw){
                const int v_idx = (((batch_idx * height + xh) * width + xw) * ${channels} + c);
                atomicAdd(&d_value[v_idx], attn[a_idx] * d_out[index]);
                ++a_idx;
            }
        }
    }
}
'''

kernel_loop = '''
#include <cupy/carray.cuh>
#include <cupy/atomics.cuh>
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import cupy
from collections import namedtuple
from string import Template

CUDA_NUM_THREADS = 1024
Stream = namedtuple('Stream', ['ptr'])

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

def Dtype(t):
    if isinstance(t, torch.cuda.HalfTensor):
        return 'float16'
    elif isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = kernel_loop + code
    code = Template(code).substitute(**kwargs)
    return cupy.RawKernel(code,kernel_name)

class nh_attn_function_set():
    def __init__(self,num_heads, channels,window_size):
        self.opt = dict(num_heads=num_heads,channels=channels,window_size=window_size,
                   window_sq_size = window_size**2, bias_size = (2*window_size-1),
                   neighborhood_size= window_size//2)
        self.kernel = {}
        self.kernel['nh_attn_forward_q_k'] = self.load_kernel_detypes('nh_attn_forward_q_k', nh_attn_forward_q_k, self.opt)
        self.kernel['nh_attn_backward_query'] = self.load_kernel_detypes('nh_attn_backward_query', nh_attn_backward_query, self.opt)
        self.kernel['nh_attn_backward_key'] = self.load_kernel_detypes('nh_attn_backward_key', nh_attn_backward_key, self.opt)
        self.kernel['nh_attn_backward_bias'] = self.load_kernel_detypes('nh_attn_backward_bias', nh_attn_backward_bias, self.opt)
        self.kernel['nh_attn_forward_attn_v'] = self.load_kernel_detypes('nh_attn_forward_attn_v', nh_attn_forward_attn_v, self.opt)
        self.kernel['nh_attn_backward_attn'] = self.load_kernel_detypes('nh_attn_backward_attn', nh_attn_backward_attn, self.opt)
        self.kernel['nh_attn_backward_value'] = self.load_kernel_detypes('nh_attn_backward_value', nh_attn_backward_value, self.opt)
        
    def __call__(self,mode,dtype,inputs):
        self.kernel[mode][dtype](**inputs)
        
    def load_kernel_detypes(self,kernel_name, code, opt):
        kernel_dict = {}
        opt['Dtype'] = 'float16'        
        kernel_dict['float16'] = load_kernel(kernel_name, code, **opt)
        opt['Dtype'] = 'float'
        kernel_dict['float'] = load_kernel(kernel_name, code, **opt)
        opt['Dtype'] = 'double'
        kernel_dict['double'] = load_kernel(kernel_name, code, **opt)
        return kernel_dict     
    
        
class nh_attn_q_k(torch.autograd.Function):    
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, bias,function_set):
        query = query.contiguous()
        key = key.contiguous()
        bias = bias.contiguous()

        assert query.dim() == 5 and query.is_cuda
        assert key.dim() == 5 and key.is_cuda
        assert bias.dim() == 3 and bias.is_cuda

        batch_size, num_heads, height, width ,channels = query.size()
        attn = torch.zeros(batch_size, num_heads, height, width,function_set.opt['window_sq_size'],dtype=query.dtype,device=query.device)

        with torch.cuda.device_of(query):
            n = attn.numel()
            inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                          grid=(GET_BLOCKS(n),1,1),
                          args=[query.data_ptr(), key.data_ptr(), bias.data_ptr(), attn.data_ptr(),
                                n,height,width],
                          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            function_set('nh_attn_forward_q_k', Dtype(query), inputs)

        ctx.save_for_backward(query, key, bias)
        ctx.function_set = function_set
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, d_attn):
        d_attn = d_attn.contiguous()

        assert d_attn.is_cuda

        query, key, bias = ctx.saved_tensors
        function_set = ctx.function_set

        batch_size, num_heads, height, width ,channels = query.size()
        d_query, d_key, d_bias = None, None, None

        with torch.cuda.device_of(d_attn):
            if ctx.needs_input_grad[0]:
                d_query = torch.zeros_like(query)
                n = d_query.numel()
                inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                              grid=(GET_BLOCKS(n),1,1),
                              args=[key.data_ptr(), d_attn.data_ptr(), d_query.data_ptr(),
                                    n,height,width],
                              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                function_set('nh_attn_backward_query',Dtype(d_query), inputs)

            if ctx.needs_input_grad[1]:
                d_key = torch.zeros_like(key)
                n = d_key.numel()
                inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                              grid=(GET_BLOCKS(n),1,1),
                              args=[query.data_ptr(), d_attn.data_ptr(), d_key.data_ptr(),
                                    n,height,width],
                              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                              
                function_set('nh_attn_backward_key',Dtype(d_key), inputs)

            if ctx.needs_input_grad[2]:
                d_bias = torch.zeros_like(bias)
                n = d_attn.numel()
                inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                              grid=(GET_BLOCKS(n),1,1),
                              args=[d_attn.data_ptr(), d_bias.data_ptr(),
                                    n,height,width],
                              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                function_set('nh_attn_backward_bias',Dtype(d_bias), inputs)
        return d_query, d_key, d_bias, None

class nh_attn_attn_v(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx,attn,value,function_set):
        attn = attn.contiguous()
        value = value.contiguous()

        assert attn.dim() == 5 and attn.is_cuda
        assert value.dim() == 5 and value.is_cuda

        batch_size, num_heads, height, width ,channels = value.size()
        out = torch.zeros_like(value)

        with torch.cuda.device_of(attn):
            n = out.numel()
            inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                          grid=(GET_BLOCKS(n),1,1),
                          args=[attn.data_ptr(), value.data_ptr(), out.data_ptr(),
                                n,height,width],
                          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            function_set('nh_attn_forward_attn_v',Dtype(out), inputs)
        ctx.save_for_backward(attn, value)
        ctx.function_set = function_set
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, d_out):
        d_out = d_out.contiguous()

        assert d_out.is_cuda

        attn, value = ctx.saved_tensors
        function_set = ctx.function_set

        batch_size, num_heads, height, width ,channels = value.size()
        d_attn, d_value = None, None

        with torch.cuda.device_of(d_out):
            if ctx.needs_input_grad[0]:
                d_attn = torch.zeros_like(attn)
                n = d_attn.numel()
                inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                              grid=(GET_BLOCKS(n),1,1),
                              args=[value.data_ptr(), d_out.data_ptr(), d_attn.data_ptr(),
                                    n,height,width],
                              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                function_set('nh_attn_backward_attn',Dtype(d_attn), inputs)

            if ctx.needs_input_grad[1]:
                d_value = torch.zeros_like(value)
                n = d_value.numel()
                inputs = dict(block=(CUDA_NUM_THREADS,1,1),
                              grid=(GET_BLOCKS(n),1,1),
                              args=[attn.data_ptr(), d_out.data_ptr(), d_value.data_ptr(),
                                    n,height,width],
                              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                function_set('nh_attn_backward_value',Dtype(d_value), inputs)

        return d_attn, d_value,None
    
class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert kernel_size > 1 and kernel_size % 2 == 1, f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        
        self.fuction_set = nh_attn_function_set(num_heads, self.head_dim, kernel_size)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens
        qkv = self.qkv(x).view(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = self.nh_attn(q,k,mode='q_k')
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.nh_attn(attn,v,mode='attn_v')
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        return self.proj_drop(self.proj(x))

    def nh_attn(self,input_1, input_2,mode='q_k'):
        if input_1.is_cuda and input_2.is_cuda and self.rpb.is_cuda:
            if mode.lower() == 'q_k':
                attn = nh_attn_q_k.apply(input_1, input_2,self.rpb,self.fuction_set)
                return attn
            elif mode.lower() == 'attn_v':
                out = nh_attn_attn_v.apply(input_1, input_2,self.fuction_set)
                return out
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

model_urls = {
    "nat_mini_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_base.pth",
}


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x
        return self.downsample(x)


class NAT(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 depths,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 kernel_size=7,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale)
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x).flatten(1, 2)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def nat_mini(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16], embed_dim=64, mlp_ratio=3,
                drop_path_rate=0.2, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_mini_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_tiny(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[2, 4, 8, 16], embed_dim=64, mlp_ratio=3,
                drop_path_rate=0.2, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_small(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[3, 6, 12, 24], embed_dim=96, mlp_ratio=2,
                drop_path_rate=0.3, layer_scale=1e-5, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_base(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[4, 8, 16, 32], embed_dim=128, mlp_ratio=2,
                drop_path_rate=0.5, layer_scale=1e-5, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
  
def test():
    model = nat_mini(True).cuda()
    model = model.eval()
    img = torch.ones(32,3,224,224).cuda()
    model(img)
    
if __name__ == '__main__' :
    if torch.cuda.is_available():
        test()
