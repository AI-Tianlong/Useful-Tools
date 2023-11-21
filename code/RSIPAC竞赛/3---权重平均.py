import os
from argparse import ArgumentParser
import torch
import time



model_dir = './work_dirs/mask2former_beit_batch2_4w'
pth_name1 = 'iter_4000.pth@iter_8000.pth@iter_12000.pth@iter_16000.pth@iter_20000.pth'

pth_name2 = 'iter_24000.pth@iter_28000.pth@iter_32000.pth@iter_36000.pth@iter_40000.pth'
pth_Final = 'swa_1.pth@swa_2.pth'

swa_name1 = 'swa_1.pth'
swa_name2 = 'swa_2.pth'
swa_Final = 'swa_Final.pth'

def swa(pth_name,swa_name):
    """swa weight average
    
    Args:
        pth_name (str): pth name, split by '@'
        swa_name (str): swa name
        
    Save:
        swa.pth
    """

    pth_list = pth_name.split("@")

    model_dirs = [                     
        os.path.join(model_dir, i)
        for i in pth_list
    ]

    print(model_dirs)
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()
    state_dict = models[-1]['state_dict']
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model['state_dict'] = new_state_dict

    save_dir = os.path.join(model_dir,swa_name)

    torch.save(ref_model, save_dir)
    print('Model is saved at', save_dir)


if __name__ == '__main__':

    print('\n')
    print(f'-----------第 3 步  权重平均---------------')
    print('\n')
    time.sleep(5)

    swa(pth_name1,swa_name1)   #swa_1.pth
    swa(pth_name2,swa_name2)   #swa_2.pth
    swa(pth_Final,swa_Final)   #swa_Final.pth

    print('\n')
    print(f'---第 3 步 已完成')
    time.sleep(5)

    print('\n')
    print(f'-----------第 4 步  测试集预测---------------')

