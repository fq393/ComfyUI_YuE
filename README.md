# ComfyUI_YuE

[![GitHub](https://img.shields.io/github/license/smthemex/ComfyUI_YuE)](https://github.com/smthemex/ComfyUI_YuE/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/smthemex/ComfyUI_YuE)](https://github.com/smthemex/ComfyUI_YuE/stargazers)

[YuE](https://github.com/multimodal-art-projection/YuE) is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). This ComfyUI integration allows you to use YuE models directly in your ComfyUI workflows.

[YuE](https://github.com/multimodal-art-projection/YuE) 是一个突破性的开源音乐生成基础模型系列，专门用于将歌词转换为完整歌曲（lyrics2song）。此 ComfyUI 集成允许您在 ComfyUI 工作流中直接使用 YuE 模型。

## 🆕 Recent Updates | 最新更新

- ✨ **Custom Audio Reference**: Added support for custom audio input files as reference prompts
- 🔧 **Quantization Libraries**: Install quantization modules (mmgp and exllamav2) only when needed
- 📁 **Model Path Update**: Updated to use ComfyUI's standard `models/yue` directory structure

- ✨ **自定义音频参考**: 新增支持自定义音频文件作为参考提示
- 🔧 **量化库优化**: 量化库（mmgp 和 exllamav2）仅在需要时安装
- 📁 **模型路径更新**: 更新为使用 ComfyUI 标准的 `models/yue` 目录结构

## 📦 Installation | 安装

### 1. Clone Repository | 克隆仓库

In the `./ComfyUI/custom_nodes` directory, run:
在 `./ComfyUI/custom_nodes` 目录中运行：

```bash
git clone https://github.com/smthemex/ComfyUI_YuE.git
```

### 2. Install Dependencies | 安装依赖

```bash
cd ComfyUI_YuE
pip install -r requirements.txt
```

**Notes | 注意事项:**
- `triton` is optional for potential acceleration | `triton` 库可选，可能提供加速效果
- If `descript-audiotools` requires higher torch version, use: | 如果 `descript-audiotools` 需要更高版本的 torch，使用：
  ```bash
  pip install --no-deps descript-audiotools
  ```

### 3. Optional Quantization Libraries | 可选量化库

**For MMGP quantization | MMGP 量化:**
```bash
pip install mmgp
```

**For ExLlamaV2 quantization | ExLlamaV2 量化:**
```bash
pip install exllamav2
```
Or install from source | 或从源码安装:
```bash
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
pip install .
```

## 📁 Model Setup | 模型设置

### 3.1 Core Models | 核心模型

Download from [xcodec_mini_infer](https://huggingface.co/m-a-p/xcodec_mini_infer/tree/main/final_ckpt) and [YuE-upsampler](https://huggingface.co/m-a-p/YuE-upsampler/tree/main):

从 [xcodec_mini_infer](https://huggingface.co/m-a-p/xcodec_mini_infer/tree/main/final_ckpt) 和 [YuE-upsampler](https://huggingface.co/m-a-p/YuE-upsampler/tree/main) 下载：

```
ComfyUI/models/yue/
├── ckpt_00360000.pth
├── decoder_131000.pth
└── decoder_151000.pth
```

### 3.2 Semantic Model | 语义模型

Download from [semantic_ckpts](https://huggingface.co/m-a-p/xcodec_mini_infer/tree/main/semantic_ckpts/hf_1_325000):

从 [semantic_ckpts](https://huggingface.co/m-a-p/xcodec_mini_infer/tree/main/semantic_ckpts/hf_1_325000) 下载：

```
ComfyUI/custom_nodes/ComfyUI_YuE/inference/xcodec_mini_infer/semantic_ckpts/hf_1_325000/
└── pytorch_model.bin
```

### 3.3 Language Models | 语言模型

**For High-End GPUs (RTX 4090/5090+) | 高端显卡 (RTX 4090/5090+):**

- **English | 英文**: [YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot) or [YuE-s1-7B-anneal-en-icl](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl)
- **Chinese | 中文**: [YuE-s1-7B-anneal-zh-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot)
- **Stage 2 | 第二阶段**: [YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general)

**Model Directory Structure | 模型目录结构:**
```
ComfyUI/models/yue/
├── YuE-s1-7B-anneal-en-cot/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors.index.json
│   ├── tokenizer.model
│   └── model-*.safetensors
├── YuE-s1-7B-anneal-zh-cot/
├── YuE-s1-7B-anneal-en-int/
└── YuE-s2-1B-general/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    └── tokenizer.model
```

### 3.4 Quantized Models (For Lower VRAM) | 量化模型（低显存）

**For VRAM ≤ 16GB | 显存 ≤ 16GB:**

Thanks to the community contributors | 感谢社区贡献者:

- **ExLlamaV2**: [Doctor-Shotgun](https://huggingface.co/Doctor-Shotgun/YuE-s1-7B-anneal-en-cot-exl2) by [@sgsdxzy](https://github.com/sgsdxzy)
- **Int8**: [Alissonerdx](https://huggingface.co/Alissonerdx/YuE-s1-7B-anneal-en-cot-int8) by [@alisson-anjos](https://github.com/alisson-anjos)
- **ExLlamaV2 Collection**: [Multiple models](https://huggingface.co/collections/Alissonerdx/yue-models-exllamav2-67a539be76b5225ebda95323) by [@alisson-anjos](https://github.com/alisson-anjos)
- **DeepBeepMeep**: [YuEGP](https://github.com/deepbeepmeep/YuEGP) by [@deepbeepmeep](https://github.com/deepbeepmeep)

## ⚙️ Configuration Guide | 配置指南

### 🚀 Performance Recommendations | 性能建议

| VRAM | Configuration | Quality | Speed | Notes |
|------|---------------|---------|-------|-------|
| **≥40GB** | `fp16` + optimized cache | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Ultimate setup |
| **≥24GB** | `fp16` + standard | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Best quality |
| **≤16GB** | `mmgp` or `exllamav2` | ⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| **≤16GB** | `int8` | ⭐⭐ | ⭐ | Not recommended |

### 🎯 Detailed Settings | 详细设置

**For 40GB+ VRAM | 40GB+ 显存 (Ultimate Performance | 终极性能):**
```
quantization_model: fp16
use_mmgp: false
offload_model: false
stage1_cache_size: 65536+
stage2_cache_size: 131072+
stage2_batch_size: 8
max_new_tokens: 8192+
```

**For 24GB+ VRAM | 24GB+ 显存 (Best Quality | 最佳质量):**
```
quantization_model: fp16
use_mmgp: false
prompt_end_time: 30s (for testing)
```

**For ≤16GB VRAM | ≤16GB 显存 (Memory Efficient | 内存高效):**
```
# Option 1: MMGP
quantization_model: fp16
use_mmgp: true
mmgp_profile: 2

# Option 2: ExLlamaV2 (Recommended)
quantization_model: exllamav2
use_mmgp: false
exllamav2_cache_mode: Q8
```

### 🎵 Audio Reference Feature | 音频参考功能

**NEW**: Custom audio input support! | **新功能**: 支持自定义音频输入！

**Custom Audio Inputs | 自定义音频输入:**
- `custom_audio_prompt`: Single-track audio reference | 单轨音频参考
- `custom_vocal_track`: Vocal track for dual-track mode | 双轨模式的人声轨道
- `custom_instrumental_track`: Instrumental track for dual-track mode | 双轨模式的伴奏轨道

**Usage Modes | 使用模式:**
- **Single Track | 单轨模式**: Enable `use_audio_prompt`, connect to `custom_audio_prompt`
- **Dual Track | 双轨模式**: Enable `use_dual_tracks_prompt`, connect to `custom_vocal_track` and `custom_instrumental_track`

**Fallback | 后备方案**: If no custom audio is provided, default files will be used automatically.
如果未提供自定义音频，将自动使用默认文件。

## 📝 Prompt Engineering | 提示词工程

### Official Guide | 官方指南
- [YuE Prompt Engineering Guide](https://github.com/multimodal-art-projection/YuE?tab=readme-ov-file#prompt-engineering-guide)
- [Top 200 Tags Reference](https://github.com/smthemex/ComfyUI_YuE/blob/main/top_200_tags.json)

### Parameter Explanation | 参数说明

- **max_new_tokens**: Controls generation length and quality (2944-16384)
  - ~0.02 seconds per token | 每个 token 约 0.02 秒
  - Higher values = longer, more coherent music | 更高的值 = 更长、更连贯的音乐

- **mmgp_profile**: Memory mapping profiles (0-4)
  - Higher values = less VRAM, more RAM, slower speed | 更高的值 = 更少显存、更多内存、更慢速度

## 🖼️ Examples | 示例

### FP16 with MMGP | FP16 配合 MMGP
![FP16 Example](https://github.com/smthemex/ComfyUI_YuE/blob/main/example.png)

### Int8 Quantization | Int8 量化
![Int8 Example](https://github.com/smthemex/ComfyUI_YuE/blob/main/int8_example.png)

## 🤝 Contributing | 贡献

We welcome contributions! Please feel free to submit issues and pull requests.
欢迎贡献！请随时提交问题和拉取请求。

## 📄 Citation | 引用

```bibtex
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```

## 📜 License | 许可证

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
本项目根据 [LICENSE](LICENSE) 文件中指定的条款进行许可。

---

**Star ⭐ this repo if you find it useful! | 如果觉得有用请给个星标！**

