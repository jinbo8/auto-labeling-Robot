# 1.参考文档

千问视觉大模型官网
https://github.com/QwenLM/Qwen2.5-VL 
模型下载路径：https://modelscope.cn/home
vllm下载官方：https://vllm.ai/
api.openai platform: https://developers.openai.com/api/docs/guides/conversation-state
更详细的环境安装策略详见文档：https://docs.vllm.ai/en/latest/getting_started/installation/index.html

vllm支持的模型列表
● Text-Only 大模型列表：https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models
● 多模态大模型列表：https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models



# 2.环境安装
conda create -n vllm python==3.9
conda activate vllm
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install modelscope transformers qwen-vl-utils[decord] accelerate vllm openai  -i https://pypi.tuna.tsinghua.edu.cn/simple


pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install qwen-vl-utils[decord] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate vllm -i https://pypi.tuna.tsinghua.edu.cn/simple


# 支持量化
pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple  #加速生成

# 3.本地部署
本地部署：
参考：https://blog.csdn.net/FL1623863129/article/details/139693141
python -m vllm.entrypoints.openai.api_server --model qwen-3B  --served-model-name Qwen2-3B-Instruct --max-model-len=2048


# 4.本地代码调用：
# python -m vllm.entrypoints.openai.api_server --model Qwen2-3B --served-model-name Qwen2-3B-Instruct --max-model-len=2048
from openai import OpenAI
client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",  # v1不可省略
    api_key="sk-1d99708bbb9c192", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen2-3B-Instruct",
  messages=[
    {"role": "user", "content": "程序员找工作面试问为什么离职怎么回答？"}
  ]
)

print(completion.choices[0].message)


# 5. api调用 代码示例
# sk-1d99708bb8714ce9b1cd3575bbb9c194

import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-1d99708bb8714ce9b1cd3575bbb9c194",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {"type": "text", "text": "对图片进行标注，标注e结果以json格式返回"},
            ],
        },
    ],
)

print(completion.choices[0].message.content)



from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["/home/jin/item/bigmodel/LV/data/1.jpg", "/home/jin/item/bigmodel/LV/data/2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk




# 6.模型权重参数下载
https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct/files
modelscope download --model qwen/Qwen2.5-VL-3B-Instruct

依赖安装：
pip install autoawq -i https://pypi.tuna.tsinghua.edu.cn/simple
# 量化权重参数下载
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ README.md --local_dir ./dir



# 7.llama_factory
https://github.com/hiyouga/LLaMA-Factory
pip install transformers==4.50.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets==3.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==1.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install peft==0.15.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install trl==0.9.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install deepspeed==0.16.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bitsandbytes==0.43.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm==0.8.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple


# 8. 自己vllm部署的大模型调用示例
https://www.bilibili.com/video/BV1Q9sAzgE7d/?spm_id_from=333.337.search-card.all.click&vd_source=c4cad7e77b4b310fcd104e46ec369d92

# 模型启动
pip install modelscope
cd 部署的模型所在路径
conda activate 虚拟环境

# 说明：可将 Qwen/Qwen3-VL-4B-Thinking-FP8 替换为自己想要部署的版本
modelscope download --model Qwen/Qwen3-VL-4B-Thinking-FP8 --local_dir Qwen/Qwen3-VL-4B-Thinking-FP8
#启动部署命令
vllm serve ./Qwen/Qwen3-VL-4B-Thinking-FP8 \
    --async-scheduling \
    --host 0.0.0.0 \
    --port 8000
    --max_model_len 8000


# 9.api调用 代码示例
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                }
            },
            {
                "type": "text",
                "text": "图片中有什么文字"
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Thinking-FP8",
    messages=messages,  # ← 这里漏掉了！
    max_tokens=2048
)

print(f"耗时: {time.time() - start:.2f}秒")
print(response.choices[0].message.content)
