import itertools
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from functools import partial

from myquant.measure.cosine import torch_cosine_similarity
from myquant.measure.updater import CosineUpdater, EQCosineUpdater, MSEUpdater
from myquant.quantizer import *
# 用于获取 activation 的最大绝对值： act_scales，维度 [in_channels]，便于后续在 W8A8SmoothConv2d 中计算 smooth_scales
class AWQObserver(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False,
        padding_mode='zeros',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.num_flag = 0

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                dtype=torch.float32,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    self.out_channels, dtype=torch.float32, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.register_buffer(
            "act_scales",
            torch.zeros(
                self.in_channels, dtype=torch.float32, requires_grad=False
            ),
        )
        # self.org_outs=[] #收集calibnum个输出用于后续计算scale
        self.inputs=[]#收集calibnum个输入用于后续计算scale
        self.tmp_num=0
    def to(self, *args, **kwargs):
        super(AWQObserver, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def forward(self, x):
        """
        observer在前向传播过程中获取信息，写入self.attribute，返回值并不重要
        """
        x_in_channels_absmax = x.abs().amax(dim=(0,2,3)) 
        self.act_scales = torch.max(self.act_scales.to(x.device), x_in_channels_absmax) # 取所有input的中的absmax

        # x_in_channels_absmean = x.abs().mean(dim=(0,2,3)) #对于[1,32,320,320]的输入，每320*320算一个平均
        # tmp_num=x.shape[2]*x.shape[3]
        # # # 取所有input的中的absmean
        # self.act_scales = (self.act_scales.to(x.device)*self.tmp_num+
        #                    x_in_channels_absmean*tmp_num)/(self.tmp_num+tmp_num) if self.tmp_num!=0 else x_in_channels_absmean
        # self.tmp_num+=tmp_num
        
        y = torch.functional.F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.inputs.append(x)
        return y
    
    @staticmethod
    def to_calib_model(
        module
    ):
        """
        把原来的conv2d模块换成calib
        """
        assert isinstance(module, torch.nn.Conv2d)
        new_module = AWQObserver(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        new_module.weight = module.weight

        if module.bias is not None:
            new_module.bias = module.bias
        return new_module
    
    def __repr__(self):
        return f"AWQObserver({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"


class W4A16AWQConv2d(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False,
        padding_mode='zeros',
        act_quant="per_tensor",
        quantize_output=False,
        a_num_of_bit=8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                dtype=torch.float32,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    self.out_channels, dtype=torch.float32, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.register_buffer(
            "smooth_scales",
            torch.ones(
                self.in_channels, dtype=torch.float32, requires_grad=False
            ),
        )
        self.act_quant_name=None
        self.act_quant=None
        self.output_quant_name=None
        self.output_quant=None

    def to(self, *args, **kwargs):
        super(W4A16AWQConv2d, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        s_x = x/self.smooth_scales.view(1,-1,1,1).to(x.device) # smooth x/s

        q_x = self.act_quant(s_x)
        y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_channel", quantize_output=False,w_num_of_bit=8,a_num_of_bit=8
    ):
        assert isinstance(module, AWQObserver)
        new_module = W4A16AWQConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_num_of_bit=a_num_of_bit
        )
        
        
        if act_quant == "per_channel":
            act_quant_method = partial(quantize_activation_per_channel_absmax, n_bits=a_num_of_bit,dim=1)
        elif act_quant == "per_channel_asym":
            act_quant_method = partial(quantize_activation_per_channel_minmax_asym, n_bits=a_num_of_bit,dim=1)
        elif act_quant == "per_tensor":
            act_quant_method = partial(quantize_activation_per_tensor_absmax, n_bits=a_num_of_bit)
        elif act_quant == "per_tensor_percentile":
            act_quant_method = partial(quantize_activation_per_tensor_percentile, n_bits=a_num_of_bit)
        elif act_quant == "per_tensor_asym":
            act_quant_method = partial(quantize_activation_per_tensor_minmax_asym, n_bits=a_num_of_bit)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")
        
        new_module.act_quant=act_quant_method
        new_module.act_quant_name=act_quant

        if quantize_output:
            new_module.output_quant_name = new_module.act_quant_name
            new_module.output_quant = new_module.act_quant
        else:
            new_module.output_quant_name = "None"
            new_module.output_quant = lambda x: x

        weight_scales = module.weight.abs().amax(dim=(0,2,3))
        # awq方案
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        x_max = module.act_scales
        n_grid = 20
        history = []

        # def f(s):
        #     # 示例目标函数，假设是一个简单的二次函数
        #     return np.sum((s - 5)**2)

        # scales=torch.rand_like(x_max)
        # # 梯度下降
        # learning_rate = 0.01
        # num_iterations = 1000
        # for _ in range(num_iterations):
        #     s -= learning_rate * gradient_f(s)

        # best_s = s
        # best_f_value = f(s)
        in_channels=module.weight.shape[1]
        # 创建 0.1 到 0.9 之间的 9 个值
        s1_part1 = np.arange(0.02, 1, 0.02)
        # 创建 1 到 10 之间的 10 个值
        s1_part2 = np.arange(1, 10, 0.2)
        # 合并两个部分
        # s1_range = np.concatenate((s1_part1, s1_part2))
        # s1_range = s1_part1
        s1_range = s1_part2
        # s1_range=np.arange(0.1, 10.0, 0.5)
        # 全channodeel搜索
        # s1_combinations = list(itertools.product(s1_range, repeat=in_channels))
        # 只搜索前两个channel
        s1_combinations=[tuple(list(i)+[1 for i in range(in_channels-2)]) for i in itertools.product(s1_range, repeat=2)]
        # updater=MSEUpdater()
        # updater=EQCosineUpdater()

        # 顺序搜索，贪心算法，优化完第一个再优化第二个
        # search_order=[i for i in range(in_channels)]
        # # search_order=[i for i in range(in_channels-1,-1,-1)]
        # best_scales=[1 for i in range(in_channels)]
        # for channel in search_order:
        #     for scale in s1_range:
        #         tmp_scale=best_scales.copy()
        #         tmp_scale[channel]=scale
        #         scales=torch.tensor(tmp_scale,dtype=torch.float32).to(module.weight.device)
        #         new_weight=module.weight.clone()
        #         new_weight=new_weight.mul_(scales.view(1,-1,1,1).to(module.weight.device)) # w · s
        #         if weight_quant == "per_channel":
        #             new_weight.data = quantize_weight_per_channel_absmax(new_weight.data,n_bits=w_num_of_bit,dim=1) # Q(w*s)
        #         elif weight_quant == "per_tensor":
        #             new_weight.data = quantize_weight_per_tensor_absmax(new_weight.data,n_bits=w_num_of_bit) # Q(w*s)
        #         updater=MSEUpdater()
        #         for i in range(len(module.inputs)):
        #             org_out = torch.functional.F.conv2d(module.inputs[i], module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #             new_out = torch.functional.F.conv2d(
        #                 act_quant_method(module.inputs[i]/scales.view(1,-1,1,1).to(module.weight.device),n_bits=a_num_of_bit), new_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #             updater.update(y_pred=new_out,y_real=org_out)
        #         loss=updater.data
        #         item=torch.cat((scales,loss.reshape(1))).cpu()
        #         history.append(item)
        #         is_best = loss < best_error
        #         if is_best:
        #             best_error = loss
        #             best_scales[channel] = tmp_scale[channel]
        # best_scales=torch.tensor(best_scales,dtype=torch.float32).to(module.weight.device)


        # easyquant交替搜索
        # smooth方案，基于计算的，作为初始化scale
        weight_scales = module.weight.abs().amax(dim=(0,2,3))
        best_scales = x_max**0.5 / weight_scales**0.5
        # 1初始化
        # best_scales = torch.ones_like(best_scales)
        if torch.isnan(best_scales).any() or torch.isinf(best_scales).any():
            best_scales = torch.ones_like(best_scales)
        print(f"迁移尺度初始化{best_scales}")
        assert isinstance(best_scales,torch.Tensor)
        # 超参数
        alpha=0.5
        beta=2
        n=20
        max_search_iter=10
        # 秒为单位
        max_search_time=600
        search_order=[i for i in range(in_channels)]
        # search_order=[i for i in range(in_channels-1,-1,-1)]
        start_time=time.time()
        iter=0
        while True:
            iter_time=time.time()
            last_iter_best_scales=best_scales.clone()
            for channel in search_order:
                scale_search_space=np.linspace(alpha*best_scales[channel].item(),beta*best_scales[channel].item(),n)
                for scale in scale_search_space:
                    tmp_scale=best_scales.clone()
                    tmp_scale[channel]=scale
                    scales=torch.tensor(tmp_scale,dtype=torch.float32).to(module.weight.device)
                    new_weight=module.weight.clone()
                    new_weight=new_weight.mul_(scales.view(1,-1,1,1).to(module.weight.device)) # w · s
                    if weight_quant == "per_channel":
                        new_weight.data = quantize_weight_per_channel_absmax(new_weight.data,n_bits=w_num_of_bit,dim=1) # Q(w*s)
                    elif weight_quant == "per_tensor":
                        new_weight.data = quantize_weight_per_tensor_absmax(new_weight.data,n_bits=w_num_of_bit) # Q(w*s)
                    updater=MSEUpdater()
                    for i in range(len(module.inputs)):
                        org_out = torch.functional.F.conv2d(module.inputs[i], module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
                        new_out = torch.functional.F.conv2d(
                            act_quant_method(module.inputs[i]/scales.view(1,-1,1,1).to(module.weight.device),n_bits=a_num_of_bit), new_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
                        updater.update(y_pred=new_out,y_real=org_out)
                    loss=updater.data
                    # item=torch.cat((scales,loss.reshape(1))).cpu()
                    # history.append(item)
                    is_best = loss < best_error
                    if is_best:
                        best_error = loss
                        best_scales[channel] = tmp_scale[channel]
            
            if all(last_iter_best_scales==best_scales):
                print(f"搜索提前结束，陷入局部最优，总耗时{end_time-start_time}秒，最优迁移尺度为{best_scales}，best_error={best_error}")
                break
            end_time=time.time()
            iter+=1
            last_iter_best_scales
            print(f"第{iter}轮搜索结束，耗时{end_time-iter_time}秒，最优迁移尺度为{best_scales}，best_error={best_error}")
            if end_time-start_time>max_search_time or iter>max_search_iter:
                print(f"搜索结束，总耗时{end_time-start_time}秒，最优迁移尺度为{best_scales}，best_error={best_error}")
                break
        
        best_scales=torch.tensor(best_scales,dtype=torch.float32).to(module.weight.device)

        # # awq
        # # # 全局搜索
        # # for scales in s1_combinations:
        # #     scales = torch.tensor(scales,dtype=torch.float32).to(module.weight.device)
        # for ratio in range(n_grid):
        #     ratio = ratio * 1 / n_grid
        #     scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        #     scales = scales / (scales.max() * scales.min()).sqrt()
        #     new_weight=module.weight.clone()
        #     new_weight=new_weight.mul_(scales.view(1,-1,1,1).to(module.weight.device)) # w · s
        #     if weight_quant == "per_channel":
        #         # new_weight.data = quantize_weight_per_channel_absmax(new_weight.data,n_bits=w_num_of_bit,dim=1) / (scales.view(1, -1,1,1)) # Q(w*s)/s
        #         new_weight.data = quantize_weight_per_channel_absmax(new_weight.data,n_bits=w_num_of_bit,dim=1) # Q(w*s)
        #     elif weight_quant == "per_tensor":
        #         # new_weight.data = quantize_weight_per_tensor_absmax(new_weight.data,n_bits=w_num_of_bit) / (scales.view(1, -1,1,1)) # Q(w*s)/s
        #         new_weight.data = quantize_weight_per_tensor_absmax(new_weight.data,n_bits=w_num_of_bit) # Q(w*s)
        #     updater=MSEUpdater()
        #     # q_i=quantize_activation_per_channel_absmax(torch.concat(module.inputs)/)
        #     for i in range(len(module.inputs)):
        #         # org_out = module(module.inputs[i])
        #         org_out = torch.functional.F.conv2d(module.inputs[i], module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #         # x/s
        #         # new_out = torch.functional.F.conv2d(module.inputs[i], new_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #         # new_out = torch.functional.F.conv2d(module.inputs[i]/scales.view(1,-1,1,1).to(module.weight.device), new_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #         # Q(x/s) conv Q(w*s)
        #         new_out = torch.functional.F.conv2d(
        #             act_quant_method(module.inputs[i]/scales.view(1,-1,1,1).to(module.weight.device),n_bits=a_num_of_bit), new_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
        #         # loss += (
        #         #     (org_out - new_out).float().pow(2).mean().item()
        #         # )  # float prevents overflow
        #         updater.update(y_pred=new_out,y_real=org_out)
        #     # loss/=len(module.inputs)
        #     loss=updater.data
        #     # if all(scales[2:]==1):
        #     #     item=torch.cat((scales[:2],loss.reshape(1))).cpu()
        #     #     history.append(item)
        #     is_best = loss < best_error
        #     if is_best:
        #         best_error = loss
        #         # best_ratio = ratio
        #         best_scales = scales
        
        # history=torch.stack(history)
        # # # 创建fig目录
        # base_dir = 'scalefig5'
        # if not os.path.exists(base_dir):
        #     os.makedirs(base_dir)
        # # 创建一个三维图
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # X1,X2,Z=history[:,0],history[:,1],history[:,2]
        # # 绘制三维图
        # n=len(s1_range)
        # ax.plot_surface(X1.reshape(n,n), X2.reshape(n,n), Z.reshape(n,n), cmap='viridis')
        # # ax.scatter(X1, X2, Z)

        # ax.set_title('3D Plot of Loss Function (s1_channel[2:]==1)')
        # ax.set_xlabel('s1_channel1')
        # ax.set_ylabel('s1_channel2')
        # ax.set_zlabel('Loss')
        # name="input_"+"_".join([str(i) for i in list(module.inputs[0].shape)])+"_weight_"+"_".join([str(i) for i in list(module.weight.shape)])+".png"
        # fig_name=os.path.join(base_dir,name)
        # plt.savefig(fig_name)
        # if best_ratio == -1 and history==[]:
        #     print(history)
        #     raise Exception
        


        # print(f"best_ratio:{best_ratio}")
        # print(f"best_scales:{best_scales}")
        # best_scales = best_scales.view(-1)
        # 这里如果scale=1相当于前面啥都没干
        # best_scales = torch.clamp(best_scales,min=1)
        # best_scales = torch.ones_like(best_scales)
        # smooth方案，基于计算的
        # weight_scales = module.weight.abs().amax(dim=(0,2,3))
        # best_scales = x_max**0.5 / weight_scales**0.5
        new_module.smooth_scales=best_scales
        print(f"best_SCALES:{best_scales}")
        module.weight.data = module.weight.data*best_scales.view(1,-1,1,1) #w*s
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_num_of_bit,dim=1
            )
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_num_of_bit
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        torch.cuda.empty_cache()
        return new_module
    
    def __repr__(self):
        return f"W8A8SmoothConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"



# 先进行这步，在 calibration 中获取 act_scales
def awq_calib_model(
    model
):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            module = AWQObserver.to_calib_model(
                module
            )
            setattr(model, name, module)
        else:
            awq_calib_model(module)
    return model

# 再进行这步，在 val 中对 activation 和 weight 进行 smooth 后再量化
def awq_quantize_model(
    model, weight_quant="per_tensor", act_quant="per_tensor"
):
    for name, module in model.named_children():
        if isinstance(module, AWQObserver):
            # print("yes")
            module = W4A16AWQConv2d.from_float(
                module, weight_quant=weight_quant, act_quant=act_quant
            )
            setattr(model, name, module)
        else:
            awq_quantize_model(module, weight_quant=weight_quant,act_quant=act_quant)
    return model