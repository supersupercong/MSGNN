import os
import logging


#######################  网络参数设置　　############################################################
channel = 32
unit_num = 12
num_group_res = 4
ssim_loss = True
# eca
eca = True
# lamda
lamda = False
#################### graph
graph = True
guide = True
n_neighbors = 5
graph_patch_size = 3
gcn_stride = 3
window = 30
LEAKY_VALUE = 0.0                     # when value = 0.0, lrelu->relu
RES_SCALE = 0.1                     # 0.1 for edsr, 1 for baseline edsr
WITH_WINDOW = True
WITH_ADAIN_NROM = False
WITH_DIFF = True
WITH_SCORE = False
training = True
########################################################################################
aug_data = False # Set as False for fair comparison

patch_size = 64
pic_is_pair = True  #input picture is pair or single

lr = 0.0005

data_dir = '/data1/wangcong/dataset/rain100H'
if pic_is_pair is False:
    data_dir = '/data1/wangcong/dataset/real-world-images'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest_net')
save_steps = 400

num_workers = 8

num_GPU = 2

device_id = '0,1'

epoch = 500
batch_size = 8

if pic_is_pair:
    root_dir = os.path.join(data_dir, 'train')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(3/5 * epoch * num_datasets / batch_size)
    l2 = int(4/5 * epoch * num_datasets / batch_size)
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


