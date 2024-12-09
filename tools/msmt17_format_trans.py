import os
import random
"""
ori:

MSMT17
├── bounding_box_test
       ├── 0000_c1_0002.jpg
       ├── 0000_c1_0003.jpg
       ├── 0000_c1_0005.jpg
├── bounding_box_train
       ├── 0000_c1_0000.jpg
       ├── 0000_c1_0001.jpg
       ├── 0000_c1_0002.jpg
├── query
       ├── 0000_c1_0000.jpg
       ├── 0000_c1_0001.jpg
       ├── 0000_c14_0030.jpg
————————————————————————————
bounding_box_train->train
bounding_box_tests->test
query->query
list_train.txt
list_val.txt
list_query.txt
list_gallery.txt
"""



def generate_qg_meta(q_dir,g_dir):
    """
    applied for query and gallery
    """
    content = []
    print(f"query:{len(os.listdir(q_dir))}")
    for fn in os.listdir(q_dir):
        pid = int(fn.split('_')[0])
        content.append("{} {}".format(fn,pid))
    with open('/data/jhb_data/datasets/MSMT17/list_query.txt','w') as f:
        f.write('\n'.join(content))
        
    content_g = []
    print(f"gallery:{len(os.listdir(g_dir))}")
    for fn in os.listdir(g_dir):
        pid = int(fn.split('_')[0])
        content_g.append("{} {}".format(fn,pid))
    with open('/data/jhb_data/datasets/MSMT17/list_gallery.txt','w') as f:
        f.write('\n'.join(content_g))
    
def generate_trainval_meta(dir,ratio=0.7):
    """
    applied for train and vaild
    """
    
    content = []
    trainval_num = len(os.listdir(dir))
    print(f"train and val:{trainval_num}")
    for fn in os.listdir(dir):
        fp = os.path.join(dir,fn)
        pid = int(fn.split('_')[0])
        content.append("{} {}".format(fn,pid))
    # 随机分割图像数据集
    train_num = int(ratio*trainval_num)
    random.shuffle(content)
    train_content = content[:train_num]
    val_content = content[train_num:]
    with open('/data/jhb_data/datasets/MSMT17/list_train.txt','w') as f:
        f.write('\n'.join(train_content))
    with open('/data/jhb_data/datasets/MSMT17/list_val.txt','w') as f:
        f.write('\n'.join(val_content))

# 重命名
os.rename('/data/jhb_data/datasets/MSMT17/bounding_box_train',
          '/data/jhb_data/datasets/MSMT17/train')
os.rename('/data/jhb_data/datasets/MSMT17/bounding_box_test', 
          '/data/jhb_data/datasets/MSMT17/test')
# 生成meta(修改对应路径)
q_dir = '/data/jhb_data/datasets/MSMT17/query'
g_dir = '/data/jhb_data/datasets/MSMT17/test'
dir = '/data/jhb_data/datasets/MSMT17/train'
generate_qg_meta(q_dir,g_dir)
generate_trainval_meta(dir)



        
        
        

