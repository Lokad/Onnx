"""Separately structured NumPy and Torch float64 stem calculations."""
import math
from common import np, CONVS, GEOMETRY


def conv_numpy(x,w,bias,stride,pad,groups):
    assert x.dtype==w.dtype==bias.dtype==np.float64 and x.ndim==w.ndim==4
    n,c,h,width=x.shape;out,per_group,kh,kw=w.shape
    assert c%groups==out%groups==0 and per_group==c//groups
    padded=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(2,3))[:,:,::stride,::stride]
    oh,ow=windows.shape[2:4]
    if groups==c==out and per_group==1:
        # A direct nine-product reduction, independent of the Torch Conv kernel.
        result=np.einsum('nchwkl,ckl->nchw',windows,w[:,0],optimize=False)
    else:
        result=np.empty((n,out,oh,ow),np.float64)
        for g in range(groups):
            patch=windows[:,g*per_group:(g+1)*per_group].transpose(0,2,3,1,4,5).reshape(n*oh*ow,-1)
            weights=w[g*(out//groups):(g+1)*(out//groups)].reshape(out//groups,-1)
            product=patch@weights.T
            result[:,g*(out//groups):(g+1)*(out//groups)]=product.reshape(n,oh,ow,out//groups).transpose(0,3,1,2)
    return result+bias[None,:,None,None]


def scalar_conv(x,w,bias,coordinate,stride,pad,groups):
    n,oc,oh,ow=coordinate;channels=w.shape[1];first=(oc//(w.shape[0]//groups))*channels
    terms=[]
    for ic in range(channels):
        for ky in range(w.shape[2]):
            iy=oh*stride-pad+ky
            for kx in range(w.shape[3]):
                ix=ow*stride-pad+kx
                if 0<=iy<x.shape[2] and 0<=ix<x.shape[3]:terms.append(float(x[n,first+ic,iy,ix])*float(w[oc,ic,ky,kx]))
    return math.fsum(terms)+float(bias[oc])


def numpy_route(features,weights,capture):
    x=features.astype(np.float64).transpose(0,2,1)[:,None]
    for i in CONVS:
        w=weights[f'conv{i}.weight'].astype(np.float64);b=weights[f'conv{i}.bias'].astype(np.float64)
        stride,pad,groups=GEOMETRY[i]
        y=conv_numpy(x,w,b,stride,pad,groups)
        capture(f'conv{i}',y,x,w,b,(stride,pad,groups));x=y
        if i in (0,3,6):
            x=np.maximum(x,0.);capture(f'relu{i}',x)
    x=x.transpose(0,2,1,3).reshape(x.shape[0],x.shape[2],-1).copy();capture('reshape',x)
    w=weights['projection.weight'].astype(np.float64)
    y=x@w;capture('projection',y,x,w)
    capture('stem',y+weights['projection.bias'].astype(np.float64))


def torch_route(features,weights,capture):
    import torch
    import torch.nn.functional as F
    x=torch.from_numpy(features.copy()).to(torch.float64).permute(0,2,1).unsqueeze(1)
    for i in CONVS:
        w=torch.tensor(weights[f'conv{i}.weight'],dtype=torch.float64)
        bias=torch.tensor(weights[f'conv{i}.bias'],dtype=torch.float64)
        stride,pad,groups=GEOMETRY[i]
        y=F.conv2d(x,w,bias,stride=stride,padding=pad,groups=groups)
        capture(f'conv{i}',y.numpy(),x.numpy(),w.numpy(),bias.numpy(),(stride,pad,groups));x=y
        if i in (0,3,6):
            x=torch.relu(x);capture(f'relu{i}',x.numpy())
    x=x.permute(0,2,1,3).contiguous().flatten(2);capture('reshape',x.numpy())
    w=torch.tensor(weights['projection.weight'],dtype=torch.float64)
    y=torch.matmul(x,w);capture('projection',y.numpy(),x.numpy(),w.numpy())
    y=torch.add(torch.tensor(weights['projection.bias'],dtype=torch.float64),y);capture('stem',y.numpy())
