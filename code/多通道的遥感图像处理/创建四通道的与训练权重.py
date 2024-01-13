from osgeo import gdal
import torch
from mmcv import Config
from mmseg.apis import init_segmentor
from mmcv.runner.checkpoint import save_checkpoint, CheckpointLoader
import mmseg_custom  # noqa: F401,F403


if __name__ == '__main__':
    config_path = '/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/ATL_config/ATL_1_mask2former_beit_fusai.py'
    resume_path = '/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/pretrained/beitv2_large_patch16_224_pt1k_ft21k.pth'
    checkpoint = CheckpointLoader.load_checkpoint(resume_path, map_location='cpu')
    
    CLASSES = ('ShuiTian', 'HanDi','QiTaJianSheYongDi',
                'YoulinDi','GuanMuLin','ShuLinDi','QiTaLinDi',
                'GaoFugaiCao','ZhongFugaiCaoDi','DiFuGaiCaoDi',
                'HeQu','ShuiKu','TanDi','Chengzhenyongdi','nongcunjumindain',
                'hupo')

    PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                 [180, 165, 180], [90, 120, 150], [102, 102, 156],
                 [128, 64, 255], [140, 140, 200], [170, 170, 170],
                 [250, 170, 160], [96, 96, 96],
                 [230, 150, 140], [128, 64, 128], [110, 110, 110],
                 [244, 35, 232]]

    checkpoint['meta'] = dict()
    checkpoint['meta']['CLASSES'] = CLASSES
    checkpoint['meta']['PALETTE'] = PALETTE
    
    print('----------------------------------')
    print(checkpoint.keys())
    print('----------------------------------')
    torch.save(checkpoint,'/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/pretrained/beit3channel.pth')

    cp_path = '/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/pretrained/beit3channel.pth'

    cfg = Config.fromfile(config_path)
    print('开始处理')
    with torch.no_grad():
        # 用默认3通道读取权重数据
        cfg.model.backbone.in_chans = 3
        model = init_segmentor(cfg, cp_path, device='cuda:0')
        print(model.keys())
        # 将输入通道更改为10，再读取权重数据
        cfg.model.backbone.in_chans = 4
        model10 = init_segmentor(cfg, cp_path, device='cuda:0')
        print(model10.keys())
        # 用kaiming方法初始化10通道权重
        torch.nn.init.kaiming_normal_(model10.backbone.patch_embed.conv1.weight, mode='fan_out', nonlinearity='relu')
        # 将前3个通道权重修改为原始预训练权重
        model10.backbone.patch_embed.conv1.weight[:, :3] = model.backbone.patch_embed.conv1.weight[:, :3]
        save_checkpoint(model=model10,
                        filename='/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/pretrained/beitv2_large_patch16_224_pt1k_ft21k_4channel.pth')
        print('处理完成')
