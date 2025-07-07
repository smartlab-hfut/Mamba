import argparse
import os
import torch
from exp.train_and_test import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

def set_random_seed(seed):
    """
    设置随机种子以确保实验结果的可重复性。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='TimesNet')

    # 基本配置
    parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='任务名称：[Exp_classification]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='状态')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='模型名称：[Autoformer, Mamba, TimesNet]')

    # 数据配置
    parser.add_argument('--data', type=str, required=True, default='data', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data', help='数据文件的根目录')
    parser.add_argument('--data_path', type=str, default='data.csv', help='数据文件名')
    parser.add_argument('--seq_len', type=int, default=100, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=100, help='预测序列长度')
    parser.add_argument('--d_model', type=int, default=6, help='序列维度')
    parser.add_argument('--expand', type=int, default=2, help='扩展倍数')
    parser.add_argument('--d_ff', type=int, default=12, help='中间维度')
    parser.add_argument('--d_conv', type=int, default=4, help='卷积尺寸')
    parser.add_argument('--c_out', type=int, default=6, help='输出维度')
    parser.add_argument('--num_layers', type=int, default=1, help='SimpMamba的层数')  # 新增参数
    parser.add_argument('--num_classes', type=int, default=5, help='分类数')  # 新增参数

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点路径')

    # 优化配置
    parser.add_argument('--train_epochs', type=int, default=200, help='训练周期')
    parser.add_argument('--batch_size', type=int, default=16
                        , help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')

    return parser.parse_args()

def configure_gpu(args):
    """
    配置GPU设备。
    """
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu:
        print(f"Using GPU: {args.gpu}")
    else:
        print("Using CPU")

def main():
    # 设置随机种子
    set_random_seed(seed=42)

    # 解析命令行参数
    args = parse_arguments()

    # 配置GPU设备
    configure_gpu(args)

    # 打印参数信息
    print('Args in experiment:')
    print_args(args)

    # 根据任务选择实验类
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        raise ValueError(f"Unsupported task_name: {args.task_name}")

    # 执行实验
    setting = f"{args.task_name}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}"

    if args.is_training:
        exp = Exp(args)  # 初始化实验实例

        print(f'>>>>>>> Start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train()

        print(f'>>>>>>> Testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test()

        torch.cuda.empty_cache()
    else:
        exp = Exp(args)

        print(f'>>>>>>> Testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test()

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
