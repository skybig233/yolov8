"""
类似smoothquant中fake_quant.py
"""
import torch

from pytorch_quantization.calib.histogram import HistogramCalibrator
@torch.no_grad()

def quantize_weight_per_channel_absmax(w, n_bits=8,dim=0):
    # w: (out_features, in_features, kernel_size, kernel_size)
    dim=(0,2,3) if dim==1 else (1,2,3)
    scales = w.abs().amax(dim=dim, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features, kernel_size, kernel_size)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_activation_per_channel_absmax(a, n_bits=8,dim=1):
    # a: (batch_size, in_features, height, weight)
    scales = a.abs().max(dim=dim, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    a.div_(scales).round_().mul_(scales)
    return a

@torch.no_grad()
def quantize_activation_per_tensor_absmax(a, n_bits=8):
    # a: (batch_size, in_features, height, weight)
    scales = a.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    a.div_(scales).round_().mul_(scales)
    return a


@torch.no_grad()
def quantize_activation_per_tensor_percentile(a, n_bits=8):
    # a: (batch_size, in_features, height, weight)
    # scales = torch.quantile(a, 0.999)
    if a.numel() >= 16777216:
        n = a.numel() // 16777216
        scales = torch.quantile(a.abs().view(-1)[:16777216*n].view(n, 16777216), 0.9999,1).mean()
    else:
        scales = torch.quantile(a.abs(), 0.9999)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    a.div_(scales).round_().clamp_(min=-q_max-1,max=q_max).mul_(scales)
    return a

@torch.no_grad()
def quantize_activation_per_channel_minmax_asym(a, n_bits=8,dim=1):
    # a: (batch_size, in_features, height, weight)
    scales = a.max(dim=dim, keepdim=True)[0] - a.min(dim=dim, keepdim=True)[0]
    q_min = 0
    q_max = 2**n_bits - 1
    scales.clamp_(min=1e-5).div_(q_max - q_min)
    # zero_point = q_max - ((a.max(dim=dim, keepdim=True)[0])/scales).round()
    zero_point = ((-a.min(dim=dim, keepdim=True)[0])/scales).round()
    # print(zero_point.shape)

    # a.div_(scales).round_().add_(zero_point).sub_(zero_point).mul_(scales)
    a.div_(scales).add_(zero_point).round_().sub_(zero_point).mul_(scales)
    return a

@torch.no_grad()
def quantize_activation_per_tensor_minmax_asym(a, n_bits=8):
    # a: (batch_size, in_features, height, weight)
    scales = a.max() - a.min()
    q_min = 0
    q_max = 2**n_bits - 1
    scales.clamp_(min=1e-5).div_(q_max - q_min)
    # zero_point = q_max - (a.max()/scales).round()
    zero_point=(-a.min()/scales).round()
    a.div_(scales).add_(zero_point).round_().sub_(zero_point).mul_(scales)
    return a


@torch.no_grad()
def quantize_activation_kl(a, n_bits=8, num_bins=2048):
    HistogramCalibrator(num_bits=n_bits,axis=None)
    # Step 1: Calculate histogram
    histogram = torch.histc(a, bins=num_bins)
    bin_width = (a.max() - a.min()) / num_bins
    
    # Step 2: Compute the best threshold using KL divergence
    threshold = find_kl_threshold(histogram, num_bins, n_bits)
    
    # Step 3: Perform quantization based on the KL threshold
    scale = threshold / (2 ** (n_bits - 1) - 1)
    # Calculate quantization parameters
    min_int = -2 ** (n_bits - 1)
    max_int = 2 ** (n_bits - 1) - 1
    a = torch.clamp(a, min=-threshold, max=threshold)
    a.div_(scale).round_().mul_(scale)
    
    return a

def torch_KL_divergence(hist: torch.Tensor, ref_hist: torch.Tensor, eps=1e-30) -> float:
    if hist.ndim != 1 or ref_hist.ndim != 1: raise ValueError(
        'Only 1 dimension tensor can compute KL divergence with another tensor. '\
        f'While your input hist has dimension {hist.ndim} and ref_hist has dimension {ref_hist.ndim}')
    if len(hist) != len(ref_hist): raise ValueError(
        'Can not compute KL divergence, len(hist) != len(ref_hist')

    # here we compute KL divergence at float64 precision, make sure your hist and ref_hist are stored at cpu.
    # otherwise it might be very slow.
    return torch.dot(hist.double(), torch.log10(hist.double() + eps) - torch.log10(ref_hist.double() + eps)).item()



def find_kl_threshold(histogram, num_bins, n_bits,start_bin=128):
    # Step 2a: Initialize search range and variables
    kl_divergence = []
    end_bin = num_bins - 1
    
    for i in range(start_bin, end_bin):
        # Step 2b: Simulate quantization using the current threshold
        quantized_hist = simulate_quantization(histogram, i, num_bins, n_bits)
        
        # Step 2c: Compute KL divergence between original and quantized histograms
        kl_div = torch_KL_divergence(histogram[:i+1], quantized_hist[:i+1])
        kl_divergence.append((kl_div, i))
    
    # Step 2d: Choose the threshold that minimizes KL divergence
    _, threshold_bin = min(kl_divergence, key=lambda x: x[0])
    threshold = (threshold_bin + 0.5) * (a.max() - a.min()) / num_bins
    
    return threshold

def simulate_quantization(histogram, threshold_bin, num_bins, n_bits):
    # Simulate the quantization process
    quantized_hist = torch.zeros_like(histogram)
    scale = threshold_bin / (2 ** (n_bits - 1) - 1)
    
    for i in range(threshold_bin + 1):
        quantized_bin = int(i / scale)
        quantized_hist[quantized_bin] += histogram[i]
    
    return quantized_hist

if __name__=="__main__":
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    quantized_tensor = quantize_activation_kl(input_tensor)