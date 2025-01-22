import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def hook_fn(module, input, output,layer_name,figdir):
    weight = module.weight.data.cpu().numpy()
    if not hasattr(module,"smooth_scales"):
        input_data = input[0].data.cpu().numpy()
    else:
        input_data=module.act_quant(input[0].data/module.smooth_scales.view(1,-1,1,1)).cpu().numpy()
    # 创建每层对应的文件夹
    layer_dir = os.path.join(figdir, layer_name)
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    else:
        return
    
    # 保存数据到npy文件
    np.save(f'{layer_dir}/weight_data.npy', weight)
    np.save(f'{layer_dir}/input_data.npy', input_data)


def draw_violin_plots(data_dir, fig_dir, plot_channels=8):
    """根据记录的npy文件绘制小提琴图
    Args:
        data_dir: 数据目录
        fig_dir: 图片保存目录
        plot_channels: 绘制的通道数,默认为8
    """
    # 创建保存图片的目录
    os.makedirs(fig_dir, exist_ok=True)
    
    # 遍历data_dir下的所有层目录
    for layer_name in os.listdir(data_dir):
        layer_data_dir = os.path.join(data_dir, layer_name)
        if not os.path.isdir(layer_data_dir):
            continue
            
        # 创建该层的图片保存目录
        layer_fig_dir = os.path.join(fig_dir, layer_name)
        os.makedirs(layer_fig_dir, exist_ok=True)
            
        # 加载npy文件
        weight = np.load(f'{layer_data_dir}/weight_data.npy')
        input_data = np.load(f'{layer_data_dir}/input_data.npy') 
         
        in_channels = weight.shape[1]
        
        # 将所有通道的数据整合到一起
        weight_data = [weight[:, i, :, :].flatten() for i in range(in_channels)]
        input_data_flat = [input_data[:, i, :, :].flatten() for i in range(in_channels)]
        
        # 只取前8个通道的数据进行绘图
        num_channels_to_plot = plot_channels
        weight_violin_data = np.concatenate(weight_data[:num_channels_to_plot])
        input_violin_data = np.concatenate(input_data_flat[:num_channels_to_plot])
        w_channels = np.concatenate([[f'Channel {i}'] * len(w) for i, w in enumerate(weight_data[:num_channels_to_plot])])
        a_channels = np.concatenate([[f'Channel {i}'] * len(a) for i, a in enumerate(input_data_flat[:num_channels_to_plot])])
        plt_width = 15  # 宽度与通道数成比例，最小宽度为10
        plt_height = 6  # 固定高度
        # # 生成权重小提琴图
        plt.figure(figsize=(plt_width, plt_height))
        sns.violinplot(x=w_channels, y=weight_violin_data)
        plt.title(f'Weights Violin Plot - Layer: {layer_name}')
        # plt.xticks(rotation=45)
        plt.savefig(f'{layer_fig_dir}/weights_violin.png')
        plt.close()
        
        # # 生成输入数据小提琴图
        plt.figure(figsize=(plt_width, plt_height))
        sns.violinplot(x=a_channels, y=input_violin_data)
        plt.title(f'Input Violin Plot - Layer: {layer_name}')
        # plt.xticks(rotation=45)
        plt.savefig(f'{layer_fig_dir}/input_violin.png')
        plt.close()

def draw_violin_plots_compare(before_dir, after_dir, fig_dir, plot_channels=6):
    """根据迁移前后的数据绘制对比小提琴图
    Args:
        before_dir: 迁移前的数据目录
        after_dir: 迁移后的数据目录
        fig_dir: 图片保存目录
        plot_channels: 绘制的通道数,默认为8
    """
    # 定义颜色参数
    before_color = '#8E8BFE'
    after_color = '#FEA3A2'
    # 定义字体大小
    plt.rcParams.update({
        'font.size': 16,  # 设置全局字体大小
        # 'axes.titlesize': 14,  # 坐标轴标题字体大小
        # 'axes.labelsize': 12,  # 坐标轴标签字体大小
        'xtick.labelsize': 15,  # x轴刻度标签字体大小
        'ytick.labelsize': 15,  # y轴刻度标签字体大小
        'legend.fontsize': 15,  # 图例字体大小
        # 'figure.titlesize': 15  # 图标题字体大小
    })
    violinscale="width"
    violinwidth=0.6

    os.makedirs(fig_dir, exist_ok=True)
    
    for layer_name in os.listdir(before_dir):
        before_layer_dir = os.path.join(before_dir, layer_name)
        after_layer_dir = os.path.join(after_dir, layer_name)
        
        if not os.path.isdir(before_layer_dir) or not os.path.isdir(after_layer_dir):
            continue
            
        layer_fig_dir = os.path.join(fig_dir, layer_name)
        os.makedirs(layer_fig_dir, exist_ok=True)
            
        before_weight = np.load(f'{before_layer_dir}/weight_data.npy')
        before_input = np.load(f'{before_layer_dir}/input_data.npy')
        after_weight = np.load(f'{after_layer_dir}/weight_data.npy') 
        after_input = np.load(f'{after_layer_dir}/input_data.npy')
         
        in_channels = before_weight.shape[1]
        # 每个列表是该inchannel的数据
        before_weight_data = [before_weight[:, i, :, :].flatten() for i in range(in_channels)]
        before_input_data = [before_input[:, i, :, :].flatten() for i in range(in_channels)]
        after_weight_data = [after_weight[:, i, :, :].flatten() for i in range(in_channels)]
        after_input_data = [after_input[:, i, :, :].flatten() for i in range(in_channels)]
        
        # 计算每个通道的max(before)/max(after)比值
        ratios = []
        for i in range(in_channels):
            before_max = np.max(np.abs(before_input_data[i]))
            after_max = np.max(np.abs(after_input_data[i]))
            if after_max != 0:
                ratio = before_max / after_max
                ratios.append((i, ratio))
        
        # 筛选比值在2-5之间的通道
        valid_channels = [i for i, ratio in ratios if 1 <= ratio <= 5]
        # if layer_name=="model.model.6.cv2.conv":
        #     valid_channels=[0,1,2,3,8,9,10,11,12,13]
        
        # 如果没有满足条件的通道,使用前plot_channels个通道
        if not valid_channels:
            num_channels = min(plot_channels, in_channels)
            valid_channels = list(range(num_channels))
        else:
            num_channels = min(len(valid_channels), plot_channels)
            valid_channels = valid_channels[:num_channels]

        # 重新组织数据结构
        weight_violin_data = []
        weight_groups = []
        input_violin_data = []
        input_groups = []
        
        for i in valid_channels:
            weight_violin_data.extend(before_weight_data[i])
            weight_violin_data.extend(after_weight_data[i])
            weight_groups.extend(['Before'] * len(before_weight_data[i]))
            weight_groups.extend(['After'] * len(after_weight_data[i]))
            
            input_violin_data.extend(before_input_data[i])
            input_violin_data.extend(after_input_data[i])
            input_groups.extend(['Before'] * len(before_input_data[i]))
            input_groups.extend(['After'] * len(after_input_data[i]))

        weight_channels = []
        input_channels = []
        for i in valid_channels:
            weight_channels.extend([f'Channel {i}'] * (len(before_weight_data[i]) + len(after_weight_data[i])))
            input_channels.extend([f'Channel {i}'] * (len(before_input_data[i]) + len(after_input_data[i])))

        plt_width = 10
        plt_height = 6
        
        plt.figure(figsize=(plt_width, plt_height))
        sns.violinplot(x=weight_channels, y=weight_violin_data, hue=weight_groups,scale=violinscale,width=violinwidth,
                      palette={'Before': before_color, 'After': after_color})
        plt.title(f'YOLOv8s Weights Distribution Before/After Migration\nLayer: {layer_name}')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Channel 0', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5'])
        plt.tight_layout()
        plt.savefig(f'{layer_fig_dir}/weights_violin_compare.png')
        plt.close()
        
        plt.figure(figsize=(plt_width, plt_height))
        sns.violinplot(x=input_channels, y=input_violin_data, hue=input_groups,scale=violinscale,width=violinwidth,
                      palette={'Before': before_color, 'After': after_color})
        plt.title(f'YOLOv8s Input Distribution Before/After Migration\nLayer: {layer_name}')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Channel 0', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5'])
        plt.tight_layout()
        plt.savefig(f'{layer_fig_dir}/input_violin_compare.png')
        plt.close()



def draw_violin_plots_compare_layerwise(before_dir, after_dir, fig_dir, plot_layers=4):
    """根据迁移前后的数据绘制layerwise对比小提琴图
    Args:
        before_dir: 迁移前的数据目录
        after_dir: 迁移后的数据目录
        fig_dir: 图片保存目录
        plot_channels: 绘制的通道数,默认为2
    """
    # 定义颜色参数
    before_color = '#8E8BFE'
    after_color = '#FEA3A2'

    os.makedirs(fig_dir, exist_ok=True)
    
    # 计算每个layer的max(before)/max(after)比值
    ratios = []
    for layer_name in os.listdir(before_dir):
        before_layer_dir = os.path.join(before_dir, layer_name)
        after_layer_dir = os.path.join(after_dir, layer_name)
        
        if not os.path.isdir(before_layer_dir) or not os.path.isdir(after_layer_dir):
            continue
            
        before_input = np.load(f'{before_layer_dir}/input_data.npy')
        after_input = np.load(f'{after_layer_dir}/input_data.npy')
        
        before_max = np.max(np.abs(before_input))
        after_max = np.max(np.abs(after_input))
        if after_max != 0:
            ratio = before_max / after_max
            ratios.append((layer_name, ratio))
    
    # 筛选比值在2-5之间的layer
    valid_layers = [layer for layer, ratio in ratios if 1.5 <= ratio <= 2]
    
    # 如果没有满足条件的layer,使用前plot_layers个layer
    if not valid_layers:
        layers = list(os.listdir(before_dir))[:plot_layers]
    else:
        layers = valid_layers[:plot_layers]
    
    weight_violin_data = []
    weight_groups = []
    weight_layers = []
    input_violin_data = []
    input_groups = []
    input_layers = []
    
    for layer_name in layers:
        before_layer_dir = os.path.join(before_dir, layer_name)
        after_layer_dir = os.path.join(after_dir, layer_name)
        
        if not os.path.isdir(before_layer_dir) or not os.path.isdir(after_layer_dir):
            continue
            
        before_weight = np.load(f'{before_layer_dir}/weight_data.npy')
        before_input = np.load(f'{before_layer_dir}/input_data.npy')
        after_weight = np.load(f'{after_layer_dir}/weight_data.npy') 
        after_input = np.load(f'{after_layer_dir}/input_data.npy')
        
        # 取所有数据
        before_weight_data = before_weight.flatten()
        after_weight_data = after_weight.flatten()
        before_input_data = before_input.flatten()
        after_input_data = after_input.flatten()
        
        # 组织权重数据
        weight_violin_data.extend(before_weight_data)
        weight_violin_data.extend(after_weight_data)
        weight_groups.extend(['Before'] * len(before_weight_data))
        weight_groups.extend(['After'] * len(after_weight_data))
        weight_layers.extend([layer_name] * (len(before_weight_data) + len(after_weight_data)))
        
        # 组织输入数据
        input_violin_data.extend(before_input_data)
        input_violin_data.extend(after_input_data)
        input_groups.extend(['Before'] * len(before_input_data))
        input_groups.extend(['After'] * len(after_input_data))
        input_layers.extend([layer_name] * (len(before_input_data) + len(after_input_data)))
    
    plt_width = 10
    plt_height = 6
    
    # 绘制权重分布图
    plt.figure(figsize=(plt_width, plt_height))
    sns.violinplot(x=weight_layers, y=weight_violin_data, hue=weight_groups,
                  palette={'Before': before_color, 'After': after_color})
    plt.title('Weights Distribution Before/After Migration - Layerwise')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/weights_violin_compare_layerwise.png')
    plt.close()
    
    # 绘制输入分布图
    plt.figure(figsize=(plt_width, plt_height))
    sns.violinplot(x=input_layers, y=input_violin_data, hue=input_groups,
                  palette={'Before': before_color, 'After': after_color})
    plt.title('Input Distribution Before/After Migration - Layerwise')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/input_violin_compare_layerwise.png')
    plt.close()

if __name__ == '__main__':
    # draw_violin_plots(data_dir="before_migration_data",fig_dir="before_migration_fig")
    draw_violin_plots_compare(
        before_dir="before_migration_data",
        after_dir="after_migration_data", 
        fig_dir="migration_compare_fig"
    )