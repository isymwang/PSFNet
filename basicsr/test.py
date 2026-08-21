import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


#测试流程函数，用于对预训练模型进行评估。
def test_pipeline(root_path):
    # 解析选项，设置分布式设置，设置随机种子
    opt, _ = parse_options(root_path, is_train=False)

    # 设置CUDA的cuDNN库为性能模式
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 创建目录并初始化日志记录器
    make_exp_dirs(opt) # 创建实验所需的目录
    # 构造日志文件路径
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    # 获取根日志记录器
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # 记录环境信息
    logger.info(get_env_info())
    # 记录选项信息
    logger.info(dict2str(opt))

    # 创建测试数据集和数据加载器
    test_loaders = [] # 初始化测试数据加载器列表
    # 遍历数据集选项
    for _, dataset_opt in sorted(opt['datasets'].items()):
        # 构建测试数据集
        test_set = build_dataset(dataset_opt)
        # 构建测试数据加载器
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        # 记录测试图像数量
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        # 将测试数据加载器添加到列表
        test_loaders.append(test_loader)

    # 创建模型
    model = build_model(opt)

    # 遍历测试数据加载器
    for test_loader in test_loaders:
        # 获取测试数据集名称
        test_set_name = test_loader.dataset.opt['name']
        # 记录测试数据集名称
        logger.info(f'Testing {test_set_name}...')
        # 执行模型验证
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    # 获取项目的根目录路径
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # 调用测试流程函数
    test_pipeline(root_path)
