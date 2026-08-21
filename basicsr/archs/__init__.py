from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY


from .smfanet_arch import PSFNet


__all__ = ['build_network']


def build_network(opt):
    """
    Build network from configuration.

    Example:
        network_g:
          type: PSFNet
          dim: 36
          n_blocks: 8
          upscaling_factor: 4
    """

    # 深拷贝配置，避免修改原始 opt
    opt = deepcopy(opt)

    # 获取网络类型
    network_type = opt.pop('type')

    logger = get_root_logger()

    # 输出当前已经注册的网络，便于检查
    logger.info(
        f"Registered networks: {list(ARCH_REGISTRY._obj_map.keys())}"
    )

    # 检查网络是否已经注册
    if network_type not in ARCH_REGISTRY._obj_map:
        raise KeyError(
            f"未找到网络类型 '{network_type}'，"
            f"当前已注册类型: {list(ARCH_REGISTRY._obj_map.keys())}"
        )

    # 获取网络类
    net_class = ARCH_REGISTRY._obj_map[network_type]

    # 创建网络
    net = net_class(**opt)

    logger.info(
        f'Network [{net.__class__.__name__}] is created.'
    )

    return net