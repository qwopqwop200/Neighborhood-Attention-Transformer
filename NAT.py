import torch
from torch import nn
from torch.nn import functional as F
import math

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
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
    
class NeighborhoodAttention(nn.Module): #It can use dynamic size as input,but is relatively slower than NeighborhoodAttention_predefined.
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert window_size%2 == 1,'windowsize must be odd.'
        
        self.dim = dim
        self.window_size = window_size
        self.shift_size = self.window_size//2
        self.mid_cell = self.shift_size * 2
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.unfold = nn.Unfold(kernel_size=self.window_size)
        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size//2)
        
        self.relative_bias = nn.Parameter(torch.zeros((2*self.window_size-1)**2,num_heads))
        trunc_normal_(self.relative_bias, std=.02)
        
        self.idx_h = torch.arange(0,window_size)
        self.idx_w = torch.arange(0,window_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w).view(-1)
        
    def forward(self, x):
        B,C,H,W = x.shape        
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
                
        attn_idx = self.get_attn_idx(H,W)
        bias_idx = self.get_bias_idx(H,W)
        
        qkv = self.qkv(x).view(B, 3, C, H, W).permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #B,C,H,W
        q = q * self.scale
        k = self.unfold(k).view(B,C,-1,self.window_size**2) #B,C,L,K^2
        v = self.unfold(v).view(B,C,-1,self.window_size**2) #B,C,L,K^2
        q = q.view(B,self.num_heads,C//self.num_heads,H*W,1).contiguous().permute(0,1,3,4,2) #B,nh,H*W,1,hC
        k = k.view(B,self.num_heads,C//self.num_heads,-1,self.window_size**2).permute(0,1,3,2,4) #B,nh,L,hc,K^2
        v = v.view(B,self.num_heads,C//self.num_heads,-1,self.window_size**2).permute(0,1,3,4,2) #B,nh,L,K^2,hc
        attn = q @ k[:,:,attn_idx]
        attn = attn + self.relative_bias[bias_idx].permute(2, 0, 1).contiguous().unsqueeze(2).unsqueeze(0)
        attn = self.attn_drop(attn)
        x = (attn @ v[:,:,attn_idx]).squeeze(3).transpose(2,3).contiguous().view(B,C,H,W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_attn_idx(self,H,W):
        H = H - (self.window_size - 1)
        W = W - (self.window_size - 1)
        attn_idx = torch.arange(0,H*W,dtype=torch.float).view(1,1,H,W).contiguous()
        attn_idx = self.pad_idx(attn_idx).view(-1).type(torch.long)
        return attn_idx
    
    def get_bias_idx(self,H,W):
        num_repeat_h = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_h[self.window_size//2] = H-(self.window_size-1)
        num_repeat_w[self.window_size//2] = W-(self.window_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        return bias_idx.view(-1,self.window_size**2)
    
class NATLayer(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(dim, window_size, num_heads,qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class NeighborhoodAttention_predefined(nn.Module): #Although this is relatively faster than NeighborhoodAttention.It can only use static size as input,but you can define a new input size if you wish.
    def __init__(self,input_size, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert window_size%2 == 1,'windowsize must be odd.'
        self.dim = dim
        self.window_size = window_size
        self.shift_size = self.window_size//2
        self.mid_cell = self.shift_size * 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.unfold = nn.Unfold(kernel_size=self.window_size)
        self.qkv = nn.Conv2d(dim,dim*3,1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size//2)
        
        self.relative_bias = nn.Parameter(torch.zeros((2*self.window_size-1)**2,num_heads))
        trunc_normal_(self.relative_bias, std=.02)
        
        self.idx_h = torch.arange(0,window_size)
        self.idx_w = torch.arange(0,window_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w).view(-1)
        
        self.set_input_size(input_size)
        
    def forward(self, x):
        B,C,H,W = x.shape
        assert H == self.H and W == self.W,'input size must be same predefined input size'
        
        qkv = self.qkv(x).view(B, 3, C, H, W).permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #B,C,H,W
        q = q * self.scale
        k = self.unfold(k).view(B,C,-1,self.window_size**2) #B,C,L,K^2
        v = self.unfold(v).view(B,C,-1,self.window_size**2) #B,C,L,K^2
        q = q.view(B,self.num_heads,C//self.num_heads,H*W,1).contiguous().permute(0,1,3,4,2) #B,nh,H*W,1,hC
        k = k.view(B,self.num_heads,C//self.num_heads,-1,self.window_size**2).permute(0,1,3,2,4) #B,nh,L,hc,K^2
        v = v.view(B,self.num_heads,C//self.num_heads,-1,self.window_size**2).permute(0,1,3,4,2) #B,nh,L,K^2,hc
        attn = q @ k[:,:,self.attn_idx]
        attn = attn + self.relative_bias[self.bias_idx].permute(2, 0, 1).contiguous().unsqueeze(2).unsqueeze(0)
        attn = self.attn_drop(attn)
        x = (attn @ v[:,:,self.attn_idx]).squeeze(3).transpose(2,3).contiguous().view(B,C,H,W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_attn_idx(self,H,W):
        H = H - (self.window_size - 1)
        W = W - (self.window_size - 1)
        attn_idx = torch.arange(0,H*W,dtype=torch.float).view(1,1,H,W).contiguous()
        attn_idx = self.pad_idx(attn_idx).view(-1).type(torch.long)
        return attn_idx
    
    def get_bias_idx(self,H,W):
        num_repeat_h = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.window_size,dtype=torch.long)
        num_repeat_h[self.window_size//2] = H-(self.window_size-1)
        num_repeat_w[self.window_size//2] = W-(self.window_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.window_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        return bias_idx.view(-1,self.window_size**2)
    
    def set_input_size(self,input_size):
        H,W = input_size
        self.H,self.W = H,W
        assert H >= self.window_size and W >= self.window_size,'input size must not be smaller than window size'
        
        attn_idx = self.get_attn_idx(H,W)
        bias_idx = self.get_bias_idx(H,W)
        
        self.register_buffer("attn_idx", attn_idx)
        self.register_buffer("bias_idx", bias_idx)

class NATLayer_predefined(nn.Module):
    def __init__(self,input_size, dim, window_size, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention_predefined(input_size, dim, window_size, num_heads,qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def set_input_size(self,input_size):
        self.attn.set_input_size(input_size)
        
def test():
    model1 = NATLayer(128,3,4)
    model2 = NATLayer_predefined((3,3),128,3,4)

    img = torch.zeros(1,128,5,5)
    
    print(model1(img).shape)
    
    try:
        model2(img).shape
    except:
        print('error')
        model2.set_input_size((5,5))
        model2(img).shape
        print('success')

def test_cuda():
    model1 = NATLayer(128,3,4).cuda()
    model2 = NATLayer_predefined((3,3),128,3,4).cuda()

    img = torch.zeros(1,128,5,5).cuda()
    
    print(model1(img).shape)
    
    try:
        model2(img).shape
    except:
        print('error')
        model2.set_input_size((5,5))
        model2(img).shape
        print('success')
        
if __name__ == '__main__' :
    test()
    if torch.cuda.is_available():
        test_cuda()
