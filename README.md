<img width="523" height="739" alt="屏幕截图 2026-07-01 115403" src="https://github.com/user-attachments/assets/62162bdc-b06f-47f2-98b4-e17c58052de3" />
<img width="523" height="739" alt="屏幕截图 2026-07-01 115403" src="https://github.com/user-attachments/assets/8f1d0cc9-8ac6-41eb-9c1e-1395018a7481" />
# 🛡️ ComfyUI Binyuan · Ultimate Sampler V5.5

> An **all-in-one sampler** custom node for ComfyUI. Load a whole checkpoint *or*
> a split stack (diffusion model + dual CLIP + VAE), pick from many architectures
> (Flux / SD3 / Wan / Qwen-Image / Krea2 / Z-Image / LTXV / Hunyuan / Lumina2 …),
> set weight precision (default / fp8 / nvfp4 / bf16 …), wire LoRAs, feed
> upstream images for img2img — all from a single node.

一个节点搞定"加载模型 → 编码条件 → 采样 → 解码出图"全流程，支持整包 Checkpoint 与分离式（扩散模型+CLIP+VAE）两种加载模式，通吃主流架构。

---

## 中文说明

### 这是什么
节点 `🛡️ Binyuan采样器 V5.5`，分类 `Binyuan`。
一体化采样器：自带模型/CLIP/VAE 加载、提示词、尺寸预设、采样参数、LoRA 列表、上游图像处理，并把图像、模型、CLIP、VAE、Latent、正负条件全部作为端口输出，方便串联下游。

### 怎么用
1. 放到 `ComfyUI/custom_nodes/binyuan_sampler_plugin_v5.5/`，重启 ComfyUI。
2. 右键 → `Binyuan` → `🛡️ Binyuan采样器 V5.5`。
3. 选 `加载模式`：
   - `整包Checkpoint`：在 `Checkpoint` 下拉选一个 .safetensors 整包模型。
   - `分离式(Flux/SD3/扩散)`：分别选 `扩散模型`、`CLIP_1`/`CLIP_2`、`VAE`，并在 `CLIP_类型` 选对应架构（flux/sd3/wan/qwen_image/krea2…）。
4. 填正/负面提示词，选尺寸预设或自定义宽高，设步数/CFG/采样算法/调度器。
5. （可选）`LORA_LIST` 选 LoRA，或在 `lora_json` 里手写 JSON 列表。
6. （可选图生图）把上游图像接到 `上游图像_1`，按需设 `重绘强度`、`上游图像处理`、`Latent输入源`。
7. `图像` 端口接 `SaveImage` 出图。

### 主要输入
| 参数 | 说明 |
|---|---|
| 加载模式 | 整包Checkpoint / 分离式 |
| Checkpoint / 扩散模型 / CLIP_1 / CLIP_2 / VAE | 模型文件下拉（自动扫描对应文件夹） |
| CLIP_类型 | ACE/boogu/chroma/flux/flux2/hidream/hunyuan_image/ideogram4/krea2/lens/LTXV/lumina2/omnigen2/ovis/pid/PixArt/qwen_image/sd3/wan 等 |
| 权重精度 | default / fp8_e4m3fn / fp8_e4m3fn_fast / fp8_e5m2 / nvfp4 / pid / pixl / bf16 / fp16 |
| 串联模式 | 使用自身模型 / 继承上游模型 / 自动检测 |
| 上游图像处理 | 重新VAE编码 / 直接作为Latent / 自动选择 |
| Latent输入源 | 空Latent / 外部Latent优先 / 上游图像优先 / 上游图像拼接 |
| 尺寸助手 / 宽度 / 高度 | 预设或自定义（步长 8） |
| 步数 / CFG / Flux引导 / 采样算法 / 调度器 / 重绘强度 | 标准采样参数 |
| LORA_LIST / lora_json | 列表选 LoRA 或 JSON 写多个 |
| 自动清理显存 | 生成后清理显存 |

### 可选端口（外部直连，绕过下拉）
`外部模型`、`外部CLIP`、`外部VAE`、`外部Latent`、`外部正面条件`、`外部负面条件`、`上游图像_1/2/3`。

### 输出
`图像`(IMAGE)、`模型`(MODEL)、`CLIP`、`VAE`、`Latent`(LATENT)、`正面条件`、`负面条件`(CONDITIONING)。

### 注意
- `CLIP_类型` 必须与实际模型架构匹配，否则加载/编码会报错。
- `权重精度` 选 fp8/nvfp4 等需对应运行环境支持。
- `trending_raw.json` 是节点内"热门模型"列表的本地缓存数据，可保留或删除，不影响功能。

---

## English

### What it is
Node `🛡️ Binyuan采样器 V5.5` under category `Binyuan`.
An all-in-one sampler: loads model/CLIP/VAE, encodes prompts, runs sampling,
decodes to image — and also exposes model/CLIP/VAE/Latent/conditioning as
output ports for chaining downstream nodes.

### How to use
1. Drop into `ComfyUI/custom_nodes/binyuan_sampler_plugin_v5.5/`, restart ComfyUI.
2. Right-click → `Binyuan` → add the node.
3. Pick `加载模式` (load mode):
   - `整包Checkpoint`: choose one .safetensors checkpoint.
   - `分离式`: pick diffusion model + CLIP_1/CLIP_2 + VAE separately, and set
     `CLIP_类型` to the matching architecture (flux/sd3/wan/qwen_image/krea2…).
4. Fill prompt/negative, choose a size preset or custom W/H, set steps/CFG/
   sampler/scheduler.
5. (Optional) pick a LoRA from `LORA_LIST` or write a JSON list in `lora_json`.
6. (Optional img2img) connect upstream images to `上游图像_1`, tune `重绘强度`,
   `上游图像处理`, `Latent输入源`.
7. Connect `图像` to `SaveImage`.

### Outputs
`图像`(IMAGE), `模型`(MODEL), `CLIP`, `VAE`, `Latent`(LATENT),
`正面条件`/`负面条件`(CONDITIONING).

### Notes
- `CLIP_类型` must match the real model architecture or loading/encoding fails.
- fp8/nvfp4 weight precision requires a compatible runtime.
- `trending_raw.json` is local cache data for the in-node "trending models"
  list — safe to keep or delete.

## Files
- `__init__.py` — node `BinyuanUltimateSamplerV9` (`BinyuanUltimateSampler`)
- `js/chicun.js` — frontend size-preset helper
- `trending_raw.json` — trending-models cache data
- `README.md` / `LICENSE` (MIT)

## Install
- ComfyUI Manager: search `Binyuan Sampler`.
- Manual: `git clone https://github.com/yjitiu/ComfyUI-Binyuan-Sampler.git binyuan_sampler_plugin_v5.5`
