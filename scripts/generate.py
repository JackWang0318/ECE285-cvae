"""
使用CVAE模型生成风格迁移结果
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import CVAE
from dataset import get_data_loaders
from utils import show_images

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='使用CVAE模型生成风格迁移结果')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--gpu', action='store_true',
                        help='使用GPU')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='每个字符生成的样本数量')
    parser.add_argument('--mode', type=str, choices=['random', 'interpolation', 'transfer'], default='transfer',
                        help='生成模式: random-随机生成, interpolation-插值, transfer-风格迁移')
    parser.add_argument('--char_indices', type=int, nargs='+',
                        help='要生成的字符索引列表')
    parser.add_argument('--style_indices', type=int, nargs='+',
                        help='要使用的风格索引列表')
    
    return parser.parse_args()

def load_model(model_path, device):
    """
    加载模型
    
    参数:
        model_path (str): 模型权重文件路径
        device (torch.device): 计算设备
        
    返回:
        CVAE: 加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    args = checkpoint['args']
    
    # 创建模型
    model = CVAE(
        img_channels=1,
        img_size=28,
        latent_dim=args['latent_dim'],
        condition_dim=1355,  # TMNIST数据集中的风格数量
        hidden_dims=args.get('hidden_dims', [32, 64, 128, 256])
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    return model

def generate_random_samples(model, style_labels, device, num_samples=10, save_path=None):
    """
    生成随机样本
    
    参数:
        model (CVAE): CVAE模型
        style_labels (Tensor): 风格标签 [num_styles, condition_dim]
        device (torch.device): 计算设备
        num_samples (int): 每个风格生成的样本数量
        save_path (str): 保存路径
        
    返回:
        Tensor: 生成的样本 [num_styles * num_samples, C, H, W]
    """
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        for style in style_labels:
            # 扩展风格标签
            style_expanded = style.unsqueeze(0).repeat(num_samples, 1)
            
            # 生成样本
            samples = model.sample(num_samples, style_expanded, device)
            all_samples.append(samples)
    
    # 连接所有样本
    all_samples = torch.cat(all_samples, dim=0)
    
    # 保存结果
    if save_path:
        show_images(all_samples, title="Random Samples", save_path=save_path, nrow=num_samples)
    
    return all_samples

def generate_style_transfers(model, source_images, source_styles, target_styles, device, save_path=None):
    """
    生成风格迁移结果
    
    参数:
        model (CVAE): CVAE模型
        source_images (Tensor): 源图像 [num_chars, C, H, W]
        source_styles (Tensor): 源风格标签 [num_chars, condition_dim]
        target_styles (Tensor): 目标风格标签 [num_styles, condition_dim]
        device (torch.device): 计算设备
        save_path (str): 保存路径
        
    返回:
        Tensor: 生成的风格迁移结果 [(num_chars+1) * num_styles, C, H, W]
    """
    model.eval()
    num_chars = source_images.size(0)
    num_styles = target_styles.size(0)
    
    # 创建结果网格
    results = []
    
    # 添加原始图像作为第一行
    for i in range(num_chars):
        results.append(source_images[i])
    
    # 对每个目标风格进行迁移
    with torch.no_grad():
        for style_idx, target_style in enumerate(target_styles):
            target_style_expanded = target_style.unsqueeze(0).repeat(num_chars, 1)
            
            # 执行风格迁移
            transfers = model.reconstruct(source_images, source_styles, target_style_expanded)
            
            # 添加到结果
            for i in range(num_chars):
                results.append(transfers[i])
    
    # 转换为张量
    results = torch.stack(results)
    
    # 保存结果
    if save_path:
        show_images(results, title="Style Transfers", save_path=save_path, nrow=num_chars)
    
    return results

def generate_style_interpolations(model, source_image, source_style, target_style, device, steps=10, save_path=None):
    """
    生成风格插值结果
    
    参数:
        model (CVAE): CVAE模型
        source_image (Tensor): 源图像 [1, C, H, W]
        source_style (Tensor): 源风格标签 [1, condition_dim]
        target_style (Tensor): 目标风格标签 [1, condition_dim]
        device (torch.device): 计算设备
        steps (int): 插值步数
        save_path (str): 保存路径
        
    返回:
        Tensor: 生成的插值结果 [steps, C, H, W]
    """
    model.eval()
    
    # 创建插值权重
    alphas = torch.linspace(0, 1, steps).to(device)
    
    # 编码源图像
    with torch.no_grad():
        mu, log_var = model.encoder(source_image, source_style)
        z = model.reparameterize(mu, log_var)
        
        # 创建结果
        results = []
        
        # 对每个插值权重生成图像
        for alpha in alphas:
            # 插值风格
            interpolated_style = source_style * (1 - alpha) + target_style * alpha
            
            # 解码
            reconstruction = model.decoder(z, interpolated_style)
            results.append(reconstruction[0])
    
    # 转换为张量
    results = torch.stack(results)
    
    # 保存结果
    if save_path:
        show_images(results, title="Style Interpolations", save_path=save_path)
    
    return results

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = load_model(args.model_path, device)
    
    # 创建数据加载器
    print("创建数据加载器...")
    _, _, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # 获取一批数据
    batch = next(iter(test_loader))
    images = batch['image'].to(device)
    char_labels = batch['char_label']
    style_labels = batch['style_label'].to(device)
    
    # 如果指定了字符索引和风格索引，使用它们
    if args.char_indices:
        selected_chars = args.char_indices
    else:
        # 默认选择前5个字符
        selected_chars = list(range(min(5, len(images))))
    
    if args.style_indices:
        selected_styles = args.style_indices
    else:
        # 默认选择前5个风格
        selected_styles = list(range(min(5, len(style_labels))))
    
    # 提取选择的字符和风格
    selected_images = images[selected_chars]
    selected_char_labels = [char_labels[i] for i in selected_chars]
    selected_char_styles = style_labels[selected_chars]
    selected_target_styles = style_labels[selected_styles]
    
    # 根据模式生成结果
    if args.mode == 'random':
        print(f"生成随机样本 (每个风格 {args.num_samples} 个)...")
        samples = generate_random_samples(
            model,
            selected_target_styles,
            device,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, 'random_samples.png')
        )
        
    elif args.mode == 'transfer':
        print(f"生成风格迁移结果 ({len(selected_chars)} 个字符, {len(selected_styles)} 个风格)...")
        transfers = generate_style_transfers(
            model,
            selected_images,
            selected_char_styles,
            selected_target_styles,
            device,
            save_path=os.path.join(args.output_dir, 'style_transfers.png')
        )
        
    elif args.mode == 'interpolation':
        print("生成风格插值结果...")
        # 选择第一个字符和第一个目标风格
        source_image = selected_images[0].unsqueeze(0)
        source_style = selected_char_styles[0].unsqueeze(0)
        target_style = selected_target_styles[0].unsqueeze(0)
        
        interpolations = generate_style_interpolations(
            model,
            source_image,
            source_style,
            target_style,
            device,
            steps=10,
            save_path=os.path.join(args.output_dir, 'style_interpolations.png')
        )
    
    print(f"生成结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 