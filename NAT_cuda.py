#This is NAT implemented using cupy.
kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

nh_attn_forward_q_k = kernel_loop + '''
extern "C"
__global__ void nh_attn_forward_q_k(const ${Dtype}* query, const ${Dtype}* key, const ${Dtype}* bias, ${Dtype}* attn) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${window_seq_length};
        const int n_h = (index / ${height} / ${width}/ ${window_seq_length}) % ${num_heads};
        const int h = (index / ${width}/ ${window_seq_length}) % ${height};
        const int w = (index / ${window_seq_length}) % ${width};
        const int k = index % ${window_seq_length};
        
        const int kh = (k / ${window_size}) % ${window_size};
        const int kw = (k % ${window_size});
        
        int ph = ${shift_size};
        int pw = ${shift_size};
        int nh = h - ${shift_size};
        int nw = w - ${shift_size};
        
        if (nh < 0){
            nh = 0;
            ph = ${center_pos} - h;
        }
        else if (h + ${shift_size} >= ${height}){
            nh = ${height} - ${window_size};
            ph = ${height} - h - 1;
        }
        
        if (nw < 0){
            nw = 0;
            pw = ${center_pos} - w;
        }
        else if (w + ${shift_size} >= ${width}){
            nw = ${width} - ${window_size};
            pw = ${width} - w - 1;
        }
        
        const int q_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w) * ${channels};
        const int k_idx = (((b * ${num_heads} + n_h) * ${height} + (kh+nh)) * ${width} + (kw+nw)) * ${channels};
        const int b_idx = (n_h * ${bias_size} + (ph+kh)) * ${bias_size} + (pw+kw);
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}  && kh < ${window_size} && kw < ${window_size}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int d=0; d < ${channels}; ++d){
                update_value += query[q_idx+d] * key[k_idx+d];
            }
            update_value += bias[b_idx];
            attn[index] = update_value;
        }
    }
}
'''

nh_attn_backward_query = kernel_loop + '''
extern "C"
__global__ void nh_attn_backward_query(const ${Dtype}* const key, const ${Dtype}* const d_attn, ${Dtype}* const d_query) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${channels};
        const int n_h = (index / ${height} / ${width}/ ${channels}) % ${num_heads};
        const int h = (index / ${width}/ ${channels}) % ${height};
        const int w = (index / ${channels}) % ${width};
        const int c = index % ${channels};
        
        int nh = max(h - ${shift_size}, 0) + (h + ${shift_size} >= ${height}) * (${height} - h - ${shift_size} - 1);
        int nw = max(w - ${shift_size}, 0) + (w + ${shift_size} >= ${width}) * (${width} - w - ${shift_size} - 1);
        
        const int a_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w)*${window_seq_length};
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int kh=0, xh=nh; kh < ${window_size}; ++kh, ++xh){
                #pragma unroll
                for (int kw=0, xw=nw; kw < ${window_size}; ++kw, ++xw){
                     const int k_idx = ((((b * ${num_heads} + n_h) * ${height} + xh) * ${width} + xw) * ${channels} + c);
                     update_value += d_attn[a_idx+(kh*${window_size}+kw)] * key[k_idx];
                }
            }
            d_query[index] = update_value;
        }
    }
}
'''

nh_attn_backward_key = kernel_loop + '''
extern "C"
__global__ void nh_attn_backward_key(const ${Dtype}* const query, const ${Dtype}* const d_attn, ${Dtype}* const d_key) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${channels};
        const int n_h = (index / ${height} / ${width}/ ${channels}) % ${num_heads};
        const int h = (index / ${width}/ ${channels}) % ${height};
        const int w = (index / ${channels}) % ${width};
        const int c = index % ${channels};
        
        int nh = max(h - ${shift_size}, 0) + (h + ${shift_size} >= ${height}) * (${height} - h - ${shift_size} - 1);
        int nw = max(w - ${shift_size}, 0) + (w + ${shift_size} >= ${width}) * (${width} - w - ${shift_size} - 1);
        
        const int a_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w) * ${window_seq_length};
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int kh=0, xh=nh; kh < ${window_size}; ++kh, ++xh){
                #pragma unroll
                for (int kw=0, xw=nw; kw < ${window_size}; ++kw, ++xw){
                    const int k_idx = ((((b * ${num_heads} + n_h) * ${height} + xh) * ${width} + xw) * ${channels} + c);
                    d_key[k_idx] += query[index] * d_attn[a_idx+(kh*${window_size}+kw)];
                }
            }
        }
    }
}
'''

nh_attn_backward_bias = kernel_loop + '''
extern "C"
__global__ void nh_attn_backward_bias(const ${Dtype}* const d_attn, ${Dtype}* const d_bias) {
      CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int n_h = (index / ${height} / ${width}/ ${window_seq_length}) % ${num_heads};
        const int h = (index / ${width}/ ${window_seq_length}) % ${height};
        const int w = (index / ${window_seq_length}) % ${width};
        const int k = index % ${window_seq_length};
        
        const int kh = (k / ${window_size}) % ${window_size};
        const int kw = (k % ${window_size});
        
        int ph = ${shift_size};
        int pw = ${shift_size};
        
        if (h < ${shift_size}){
            ph = ${center_pos} - h;
        }
        else if (h + ${shift_size} >= ${height}){
            ph = ${height} - h - 1;
        }
        
        if (w < ${shift_size}){
            pw = ${center_pos} - w;
        }
        else if (w + ${shift_size} >= ${width}){
            pw = ${width} - w - 1;
        }
        
        const int b_idx = (n_h * ${bias_size} + (ph+kh)) * ${bias_size} + (pw+kw);
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}  && kh < ${window_size} && kw < ${window_size}){
            d_bias[b_idx] += d_attn[index];
        }
    }
}

'''
nh_attn_forward_attn_v = kernel_loop + '''
extern "C"
__global__ void nh_attn_forward_attn_v(const ${Dtype}* attn, const ${Dtype}* value, ${Dtype}* out) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${channels};
        const int n_h = (index / ${height} / ${width}/ ${channels}) % ${num_heads};
        const int h = (index / ${width}/ ${channels}) % ${height};
        const int w = (index / ${channels}) % ${width};
        const int c = index % ${channels};
        
        int nh = max(h - ${shift_size}, 0) + (h + ${shift_size} >= ${height}) * (${height} - h - ${shift_size} - 1);
        int nw = max(w - ${shift_size}, 0) + (w + ${shift_size} >= ${width}) * (${width} - w - ${shift_size} - 1);
        
        const int a_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w)*${window_seq_length};
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int kh=0, xh=nh; kh < ${window_size}; ++kh, ++xh){
                #pragma unroll
                for (int kw=0, xw=nw; kw < ${window_size}; ++kw, ++xw){
                     const int v_idx = ((((b * ${num_heads} + n_h) * ${height} + xh) * ${width} + xw) * ${channels} + c);
                     update_value += attn[a_idx+(kh*${window_size}+kw)] * value[v_idx];
                }
            }
            out[index] = update_value;
        }
    }
}

'''
nh_attn_backward_attn = kernel_loop + '''
extern "C"
__global__ void nh_attn_backward_attn(const ${Dtype}* const value, const ${Dtype}* const d_out, ${Dtype}* const d_attn) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${window_seq_length};
        const int n_h = (index / ${height} / ${width}/ ${window_seq_length}) % ${num_heads};
        const int h = (index / ${width}/ ${window_seq_length}) % ${height};
        const int w = (index / ${window_seq_length}) % ${width};
        const int k = index % ${window_seq_length};
        
        const int kh = (k / ${window_size}) % ${window_size};
        const int kw = (k % ${window_size});
        
        int nh = max(h - ${shift_size}, 0) + (h + ${shift_size} >= ${height}) * (${height} - h - ${shift_size} - 1);
        int nw = max(w - ${shift_size}, 0) + (w + ${shift_size} >= ${width}) * (${width} - w - ${shift_size} - 1);
        
        const int o_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w) * ${channels};
        const int v_idx = (((b * ${num_heads} + n_h) * ${height} + (nh+kh))* ${width} + (nw+kw)) * ${channels};
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}  && kh < ${window_size} && kw < ${window_size}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int d=0; d < ${channels}; ++d){
                update_value += d_out[o_idx+d] * value[v_idx+d];
            }
            d_attn[index] = update_value;
        }
    }
}
'''

nh_attn_backward_value = kernel_loop + '''
extern "C"
__global__ void nh_attn_backward_value(const ${Dtype}* const attn, const ${Dtype}* const d_out, ${Dtype}* const d_value) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int b = index / ${num_heads} / ${height} / ${width} / ${channels};
        const int n_h = (index / ${height} / ${width}/ ${channels}) % ${num_heads};
        const int h = (index / ${width}/ ${channels}) % ${height};
        const int w = (index / ${channels}) % ${width};
        const int c = index % ${channels};
        
        int nh = max(h - ${shift_size}, 0) + (h + ${shift_size} >= ${height}) * (${height} - h - ${shift_size} - 1);
        int nw = max(w - ${shift_size}, 0) + (w + ${shift_size} >= ${width}) * (${width} - w - ${shift_size} - 1);
        
        const int a_idx = (((b * ${num_heads} + n_h) * ${height} + h) * ${width} + w) * ${window_seq_length};
        
        if (h < ${height} && w < ${width} && n_h < ${num_heads}){
            ${Dtype} update_value = 0;
            #pragma unroll
            for (int kh=0, xh=nh; kh < ${window_size}; ++kh, ++xh){
                #pragma unroll
                for (int kw=0, xw=nw; kw < ${window_size}; ++kw, ++xw){
                     const int v_idx = ((((b * ${num_heads} + n_h) * ${height} + xh) * ${width} + xw) * ${channels} + c);
                     d_value[v_idx] += attn[a_idx+(kh*${window_size}+kw)] * d_out[index];
                }
            }
        }
    }
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

import cupy
from collections import namedtuple
from string import Template

CUDA_NUM_THREADS = 1024
Stream = namedtuple('Stream', ['ptr'])

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    
@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)
  
 class nh_attn_q_k(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, bias, window_size):
        assert query.dim() == 5 and query.is_cuda
        assert key.dim() == 5 and key.is_cuda
        assert bias.dim() == 3 and bias.is_cuda
        
        batch_size, num_heads, height, width ,channels = query.size()
        attn = query.new(batch_size, num_heads, height, width,window_size**2)
        
        with torch.cuda.device_of(query):
            n = attn.numel()
            opt = dict(Dtype=Dtype(query), nthreads=n,
                       batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                       window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                       center_pos=window_size-1, shift_size=window_size//2)
            f = load_kernel('nh_attn_forward_q_k', nh_attn_forward_q_k, **opt)
            
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[query.data_ptr(), key.data_ptr(), bias.data_ptr(), attn.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            
        ctx.save_for_backward(query, key, bias)
        ctx.window_size = window_size
        return attn
    
    @staticmethod
    def backward(ctx, d_attn):
        assert d_attn.is_cuda
        
        query, key, bias = ctx.saved_tensors
        window_size = ctx.window_size
        
        batch_size, num_heads, height, width ,channels = query.size()
        d_query, d_key, d_bias = None, None, None
        
        with torch.cuda.device_of(d_attn):
            if ctx.needs_input_grad[0]:
                d_query = query.new(query.size())
                n = d_query.numel()
                opt = dict(Dtype=Dtype(d_attn), nthreads=n,
                           batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                           window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                           center_pos=window_size-1, shift_size=window_size//2)
                
                f = load_kernel('nh_attn_backward_query',nh_attn_backward_query, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[key.data_ptr(), d_attn.data_ptr(), d_query.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                
            if ctx.needs_input_grad[1]:
                d_key = key.new(key.size())
                n = d_key.numel()
                opt = dict(Dtype=Dtype(d_attn), nthreads=n,
                           batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                           window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                           center_pos=window_size-1, shift_size=window_size//2)
                
                f = load_kernel('nh_attn_backward_key',nh_attn_backward_key, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[query.data_ptr(), d_attn.data_ptr(), d_key.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                
            if ctx.needs_input_grad[2]:
                d_bias = bias.new(bias.size())
                n = d_attn.numel()
                opt = dict(Dtype=Dtype(d_attn), nthreads=n,
                           batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                           window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                           center_pos=window_size-1, shift_size=window_size//2)
                
                f = load_kernel('nh_attn_backward_bias',nh_attn_backward_bias, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[d_attn.data_ptr(), d_bias.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                
        return d_query, d_key, d_bias, None

class nh_attn_attn_v(torch.autograd.Function):
    @staticmethod
    def forward(ctx,attn,value,window_size):
        assert attn.dim() == 5 and attn.is_cuda
        assert value.dim() == 5 and value.is_cuda
        
        batch_size, num_heads, height, width ,channels = value.size()
        out = value.new(batch_size, num_heads, height, width, channels)
        
        with torch.cuda.device_of(attn):
            n = out.numel()
            opt = dict(Dtype=Dtype(attn), nthreads=n,
                       batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                       window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                       center_pos=window_size-1, shift_size=window_size//2)
            
            f = load_kernel('nh_attn_forward_attn_v', nh_attn_forward_attn_v, **opt)
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[attn.data_ptr(), value.data_ptr(), out.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            
        ctx.save_for_backward(attn, value)
        ctx.window_size = window_size
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        assert d_out.is_cuda
        
        attn, value = ctx.saved_tensors
        window_size = ctx.window_size
        
        batch_size, num_heads, height, width ,channels = value.size()
        d_attn, d_value = None, None
        
        with torch.cuda.device_of(d_out):
            if ctx.needs_input_grad[0]:
                d_attn = attn.new(attn.size())
                n = d_attn.numel()
                opt = dict(Dtype=Dtype(d_out), nthreads=n,
                           batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                           window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                           center_pos=window_size-1, shift_size=window_size//2)
                
                f = load_kernel('nh_attn_backward_attn',nh_attn_backward_attn, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[value.data_ptr(), d_out.data_ptr(), d_attn.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                
            if ctx.needs_input_grad[1]:
                d_value = value.new(value.size())
                n = d_value.numel()
                opt = dict(Dtype=Dtype(d_out), nthreads=n,
                           batch=batch_size, num_heads=num_heads, height=height, width=width, channels=channels,
                           window_size=window_size, window_seq_length=window_size**2, bias_size=(2*window_size-1),
                           center_pos=window_size-1, shift_size=window_size//2)
                
                f = load_kernel('nh_attn_backward_value',nh_attn_backward_value, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[attn.data_ptr(), d_out.data_ptr(), d_value.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                
        return d_attn, d_value,None
      
class NeighborhoodAttention(nn.Module):
    def __init__(self,dim, num_heads,window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert window_size%2 == 1,'windowsize must be odd.'
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads,(2*self.window_size-1),(2*self.window_size-1)))
        
        trunc_normal_(self.relative_bias, std=.02)
        
    def forward(self, x):
        x = self.nh_attention(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def nh_attention(self,x):
        B,C,H,W = x.shape
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        qkv = self.qkv(x).view(B, 3,self.num_heads,self.head_dim,H,W).permute(1,0,2,4,5,3) # B,nh,H,W,nc
        q, k, v = qkv[0], qkv[1] ,qkv[2]
        attn = self.nh_attn(q,k,mode='q_k')
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = self.nh_attn(attn,v,mode='attn_v')
        out = out.permute(0,1,4,2,3).contiguous().view(B,C,H,W)
        return out
    
    def nh_attn(self,input_1, input_2,mode='q_k'):
        if input_1.is_cuda and input_2.is_cuda and self.relative_bias.is_cuda:
            if mode.lower() == 'q_k':
                attn = nh_attn_q_k.apply(input_1, input_2,self.relative_bias,self.window_size)
                return attn
            elif mode.lower() == 'attn_v':
                out = nh_attn_attn_v.apply(input_1, input_2,self.window_size)
                return out
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
      
class NATLayer(nn.Module):
    def __init__(self, dim, num_heads,window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, num_heads,window_size,qkv_bias, qk_scale, attn_drop, drop)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
      
def test_cuda():
    model = NATLayer(128,4).cuda()
    img = torch.rand(2,128,56,56).cuda()
    print(model(img).shape)
    
if __name__ == '__main__' :
    if torch.cuda.is_available():
        test_cuda()
