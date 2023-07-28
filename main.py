import argparse
import binascii




# 自定义类型转换函数
def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

# 创建参数解析器
parser = argparse.ArgumentParser()

parser.add_argument('--UI', type=str)
parser.add_argument('--Version', type=str)
parser.add_argument('--ControlNet', type=str_to_bool)
parser.add_argument('--Roop', type=str_to_bool)
parser.add_argument('--Drive_Map', type=str_to_bool)
parser.add_argument('--Key_words', type=str_to_bool)
parser.add_argument('--Segment', type=str_to_bool)

# 解析命令行参数
args = parser.parse_args()

UI = args.UI
Roop = args.Roop
Version = args.Version
ControlNet = args.ControlNet
Drive_Map = args.Drive_Map
Key_words = args.Key_words
Segment = args.Segment

################################################################################################################################################

import sys
import os
import base64
import importlib.util
from IPython.display import clear_output
from google.colab import drive
import tensorflow as tf

# 检测是否为GPU运行
print("TensorFlow version:", tf.__version__)
if tf.test.gpu_device_name():
    drive.mount('/content/drive')
else:
    raise Exception("\n请在《代码执行程序》-《更改运行时类型》-设置为GPU~")

# w = base64.b64decode(("d2VidWk=").encode('ascii')).decode('ascii')
# sdw = base64.b64decode(("c3RhYmxlLWRpZmZ1c2lvbi13ZWJ1aQ==").encode('ascii')).decode('ascii')
sdw = binascii.unhexlify("737461626c652d646966667573696f6e2d7765627569").decode('ascii')
w = binascii.unhexlify("7765627569").decode('ascii')
wb = f'/content/{sdw}'
gwb = f'/content/drive/MyDrive/{sdw}'

get_ipython().run_line_magic('cd', '/content')
get_ipython().run_line_magic('env', 'TF_CPP_MIN_LOG_LEVEL=1')

# 云盘同步
def cloudDriveSync(cloudPath, localPath='', sync=False):
    # 云盘没有目录
    if not os.path.exists(cloudPath):
        # 创建云盘目录
        get_ipython().system(f'mkdir {cloudPath}')
    
    # 是否要同步
    if not sync:
        return
    
    # 删除本地目录
    get_ipython().system(f'rm -rf {localPath}')
    # 链接云盘目录
    get_ipython().system(f'ln -s {cloudPath} {localPath}')
    
# 初始化云盘
def initCloudDrive():
    cloudDriveSync(f'{gwb}')
    cloudDriveSync(f'{gwb}/models')
    cloudDriveSync(f'{gwb}/lora')
    cloudDriveSync(f'{gwb}/vae')

# clong git
def gitDownload(url, localPath):
    if os.path.exists(localPath):
        return
    
    get_ipython().system(f'git clone {url} {localPath}')


# 安装附加功能
def installAdditional():
    # 预装扩展
    # gitDownload(f'https://huggingface.co/embed/lora',f'{wb}/models/Lora/positive')
    gitDownload(f'https://github.com/a2569875/{sdw}-composable-lora',f'{wb}/extensions/composable-lora')
    get_ipython().system(f'rm -rf {wb}/extensions/{sdw}-images-browser')
    gitDownload(f'https://github.com/AlUlkesh/{sdw}-images-browser',f'{wb}/extensions/{sdw}-images-browser')
    get_ipython().system(f'rm -rf {wb}/extensions/sd-civitai-browser')
    gitDownload(f'https://github.com/camenduru/sd-civitai-browser',f'{wb}/extensions/sd-civitai-browser')
    gitDownload(f'https://github.com/camenduru/sd-{w}-tunnels',f'{wb}/extensions/sd-{w}-tunnels')
    get_ipython().system(f'rm -rf {wb}/extensions/{sdw}-catppuccin')
    gitDownload(f'https://github.com/camenduru/{sdw}-catppuccin',f'{wb}/extensions/{sdw}-catppuccin')
    gitDownload(f'https://github.com/KohakuBlueleaf/a1111-sd-{w}-lycoris',f'{wb}/extensions/a1111-sd-{w}-lycoris')
    gitDownload(f'https://github.com/etherealxx/batchlinks-{w}',f'{wb}/extensions/batchlinks-{w}')
    gitDownload(f'https://github.com/AUTOMATIC1111/{sdw}-rembg', f'{wb}/extensions/{sdw}-rembg')
    gitDownload(f'https://github.com/camenduru/sd-{w}-aspect-ratio-helper', f'{wb}/extensions/sd-{w}-aspect-ratio-helper')
    # gitDownload(f'https://github.com/camenduru/sd_{w}_stealth_pnginfo', f'{wb}/extensions/sd_{w}_stealth_pnginfo')
    gitDownload(f'https://github.com/fkunn1326/openpose-editor', f'{wb}/extensions/openpose-editor')
    gitDownload(f'https://github.com/hnmr293/posex', f'{wb}/extensions/posex')
    gitDownload(f'https://github.com/nonnonstop/sd-{w}-3d-open-pose-editor', f'{wb}/extensions/sd-{w}-3d-open-pose-editor')
    gitDownload(f'https://github.com/kohya-ss/sd-{w}-additional-networks',f'{wb}/extensions/sd-{w}-additional-networks')
    gitDownload(f'https://github.com/deforum-art/deforum-for-automatic1111-{w}', f'{wb}/extensions/deforum-for-automatic1111-{w}')

    # 优化embeddings
    get_ipython().system(f'rm -rf {wb}/embeddings/negative')
    gitDownload(f'https://huggingface.co/embed/negative',f'{wb}/embeddings/negative')
    get_ipython().system(f'rm -rf {wb}/embeddings/negative/.git')
    get_ipython().system(f'rm {wb}/embeddings/negative/.gitattributes')

    #中文插件
    gitDownload(f'https://github.com/DominikDoom/a1111-sd-{w}-tagcomplete',f'{wb}/extensions/a1111-sd-{w}-tagcomplete')
    get_ipython().system(f'rm -f {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')
    get_ipython().system(f'wget https://beehomefile.oss-cn-beijing.aliyuncs.com/20210114/danbooru.csv -O {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')
    gitDownload(f'https://github.com/toriato/{sdw}-wd14-tagger',f'{wb}/extensions/{sdw}-wd14-tagge')
    get_ipython().system(f'rm -f {wb}/localizations')
    gitDownload(f'https://github.com/dtlnor/{sdw}-localization-zh_CN',f'{wb}/extensions/{sdw}-localization-zh_CN')
    #附加插件=脸部修复/漫画助手
    gitDownload(f'https://github.com/Bing-su/adetailer',f'{wb}/extensions/adetailer')
    get_ipython().system(f'wget https://huggingface.co/gmk123/mhzs/raw/main/jubenchajian4_51.py -O {wb}/scripts/jubenchajian4_51.py')
    #Roop换脸插件
    if Roop:
        gitDownload(f'https://github.com/s0md3v/sd-{w}-roop',f'{wb}/extensions/sd-{w}-roop')
        print('Roop换脸启用')
    else:
        print('Roop换脸不启用')

    # ControlNet模型
    Cnt_models = [
            'control_v11e_sd15_ip2p_fp16.safetensors',
            'control_v11e_sd15_shuffle_fp16.safetensors',
            'control_v11p_sd15_canny_fp16.safetensors',
            'control_v11f1p_sd15_depth_fp16.safetensors',
            'control_v11p_sd15_inpaint_fp16.safetensors',
            'control_v11p_sd15_lineart_fp16.safetensors',
            'control_v11p_sd15_mlsd_fp16.safetensors',
            'control_v11p_sd15_normalbae_fp16.safetensors',
            'control_v11p_sd15_openpose_fp16.safetensors',
            'control_v11p_sd15_scribble_fp16.safetensors',
            'control_v11p_sd15_seg_fp16.safetensors',
            'control_v11p_sd15_softedge_fp16.safetensors',
            'control_v11p_sd15s2_lineart_anime_fp16.safetensors',
            'control_v11f1e_sd15_tile_fp16.safetensors',
        ]
    get_ipython().system(f'rm -rf {wb}/extensions/sd-{w}-controlnet')
    # 模型下载到Colab
    if ControlNet:
        gitDownload(f'https://github.com/Mikubill/sd-{w}-controlnet',f'{wb}/extensions/sd-{w}-controlnet')
        for v in Cnt_models:
            get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/{v} -d {wb}/extensions/sd-{w}-controlnet/models -o {v}')
        print("启用 ControlNet")
    else:
        print("不启用 ControlNet")

    # 各种UI界面
    if UI == "Kitchen_Ui":
        gitDownload(f'https://github.com/canisminor1990/sd-{w}-kitchen-theme-legacy', f'{wb}/extensions/sd-{w}-kitchen-theme-legacy')
        print("Kitchen界面插件启用")
    elif UI == "Lobe_Ui":
        gitDownload(f'https://github.com/canisminor1990/sd-web-ui-kitchen-theme', f'{wb}/extensions/sd-web-ui-kitchen-theme')      
        print("Lobe界面插件启用")
    elif UI == "Ux_Ui":
        gitDownload(f'https://github.com/anapnoe/{sdw}-ux', f'{wb}/extensions/{sdw}-ux')
        print("UX界面插件启用")
    elif UI == "No":
        print("UI插件不启用")

    # 关键词
    if Key_words:
        gitDownload(f'https://github.com/Physton/sd-{w}-prompt-all-in-one', f'{wb}/extensions/sd-{w}-prompt-all-in-one')
        print("关键词插件启用")
    else:
        get_ipython().system(f'rm -rf {wb}/extensions/sd-{w}-prompt-all-in-one')
        print("关键词插件不启用")

    #下载segment插件
    if Segment:
        # 下载segment anything 插件
        print("下载segment anything 插件")
        gitDownload(f'https://github.com/continue-revolution/sd-{w}-segment-anything',
                    f'{wb}/extensions/sd-{w}-segment-anything')
        # 下载segment模型
        get_ipython().system(
            f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o {wb}/extensions/sd-{w}-segment-anything/models/sam/sam_vit_h_4b8939.pth')
    else:
        get_ipython().system(f'rm -rf {wb}/extensions/sd-{w}-segment-anything')
        print("segemnt插件不下载")
# 初始化本地环境
def initLocal():
    #部署 ubuntu3 环境
    get_ipython().system(f'apt-get -y install -qq aria2')
    get_ipython().system(f'apt -y update -qq')
    get_ipython().system(f'wget http://launchpadlibrarian.net/367274644/libgoogle-perftools-dev_2.5-2.2ubuntu3_amd64.deb')
    get_ipython().system(f'wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/google-perftools_2.5-2.2ubuntu3_all.deb')
    get_ipython().system(f'wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libtcmalloc-minimal4_2.5-2.2ubuntu3_amd64.deb')
    get_ipython().system(f'wget https://launchpad.net/ubuntu/+source/google-perftools/2.5-2.2ubuntu3/+build/14795286/+files/libgoogle-perftools4_2.5-2.2ubuntu3_amd64.deb')
    get_ipython().system(f'apt install -qq libunwind8-dev')
    get_ipython().system(f'dpkg -i *.deb')
    # get_ipython().run_line_magic('env', 'LD_PRELOAD=libtcmalloc.so')
    get_ipython().system(f'rm *.deb')

    #部署 GPU 环境
    get_ipython().system(f'apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev')
    # get_ipython().system(f'pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U')
    # get_ipython().system(f'pip install -q xformers==0.0.18 triton==2.0.0 -U')

    #主框架模块
    if Version == "A1111":
        get_ipython().system(f'git clone https://github.com/AUTOMATIC1111/{sdw} {wb}')
    elif Version == "V2.2":
        get_ipython().system(f'git clone -b v2.2 https://github.com/camenduru/{sdw} {wb}')
    elif Version == "V2.3":
        get_ipython().system(f'git clone -b v2.3 https://github.com/camenduru/{sdw} {wb}')

    get_ipython().system(f'git -C {wb}/repositories/stable-diffusion-stability-ai reset --hard')


    # 初始化云盘
    initCloudDrive()

    # 安装附加功能
    installAdditional()

    get_ipython().system(f'wget -O {wb}/config.json "https://huggingface.co/gmk123/sd_config/raw/main/config.json"')

    #映射模型、lora、图库
    if Drive_Map:
        # 创建符号链接,将gwb的模型、Lora、vae目录链接
        get_ipython().system(f'ln -s {gwb}/models {wb}/models/Stable-diffusion')
        get_ipython().system(f'ln -s {gwb}/lora {wb}/models/Lora')
        get_ipython().system(f'ln -s {gwb}/vae {wb}/models/VAE')
        print("云盘已链接")
    else:
        print("云盘不启用")
        # 如果云盘没有模型
    if len(os.listdir(f"{gwb}/models")) == 0:
        #下载主模型
        get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -d {wb}/models/Stable-diffusion -o chilloutmix_NiPrunedFp32Fix.safetensors')

    # 如果云盘Vae模型
    if len(os.listdir(f"{gwb}/vae")) == 0:
        # #VAE
        get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d {wb}/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors')
        
    #放大
    get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d {wb}/models/ESRGAN -o 4x-UltraSharp.pth')


    model_dir = os.path.join(wb, "models", "Stable-diffusion")
    if any(f.endswith(('.ckpt', '.safetensors')) for f in os.listdir(model_dir)):
        get_ipython().system(f'sed -i \'s@weight_load_location =.*@weight_load_location = "cuda"@\' {wb}/modules/shared.py')
        get_ipython().system(f'sed -i "s@os.path.splitext(model_file)@os.path.splitext(model_file); map_location=\'cuda\'@" {wb}/modules/sd_models.py')
        get_ipython().system(f'sed -i "s@map_location=\'cpu\'@map_location=\'cuda\'@" {wb}/modules/extras.py')
        get_ipython().system(f"sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' {wb}/webui.py")

    # 下载主模型
    get_ipython().system(
        f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -d {wb}/models/Stable-diffusion -o chilloutmix_NiPrunedFp32Fix.safetensors')

    get_ipython().system(
        f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/AerinK/NotSoXJB-Mix-1/resolve/main/NSX-1A-purned.safetensors -d {wb}/models/Stable-diffusion -o NSX-1A-purned.safetensors')

    get_ipython().system(
        f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/OortOnline/darkSushiMixMix_225D/resolve/main/darkSushiMixMix_225D.safetensors -d {wb}/models/Stable-diffusion -o darkSushiMixMix_225D.safetensors')
# 运行
def run(script):
    clear_output()
    get_ipython().run_line_magic('cd', f'{wb}')
    get_ipython().system(f'python {script} --listen --enable-insecure-extension-access --theme dark --gradio-queue --multiple --opt-sdp-attention --api --cors-allow-origins=*')

# 运行脚本
if os.path.exists(f'{wb}'):
    run('webui.py')
else:
    # 初化本地环境
    initLocal()
    # 运行
    run('launch.py')