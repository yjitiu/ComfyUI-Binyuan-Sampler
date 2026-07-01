import os
import re
import sys
import gc
import json
import time
import inspect
import importlib.util
import torch
import folder_paths
import nodes
import comfy.utils
import comfy.sd
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ========== 全局缓存 ==========
_FILE_CACHE = {}
_FILE_CACHE_TIMESTAMP = {}

def get_files_cached(folder):
    global _FILE_CACHE, _FILE_CACHE_TIMESTAMP
    if folder not in folder_paths.folder_names_and_paths:
        return []
    folder_paths_list = folder_paths.folder_names_and_paths[folder][0]
    cache_key = f"{folder}_{'_'.join(folder_paths_list)}"
    now = time.time()
    if cache_key in _FILE_CACHE and (now - _FILE_CACHE_TIMESTAMP.get(cache_key, 0)) < 5:
        return _FILE_CACHE[cache_key]
    files = sorted(list(set(folder_paths.get_filename_list(folder))))
    _FILE_CACHE[cache_key] = files
    _FILE_CACHE_TIMESTAMP[cache_key] = now
    return files

# ========== CLIP 类型规范化 ==========
# 将用户可见类型映射到 ComfyUI CLIPType 枚举对应名称（CLIPLoader/DualCLIPLoader 内部用 getattr(CLIPType, name.upper()) 解析）。
# pid / pixl 在 ComfyUI 中并非独立 CLIP 类型，而是 PixelDiffusion 解码器架构（comfy.ldm.pixeldit.pid），
# 其文本编码器与 PixelDiT 家族同源，因此归一化到 pixeldit，保证可正常加载。
CLIP_TYPE_CANONICAL = {
    "ace": "ace", "boogu": "boogu", "chroma": "chroma", "cogvideox": "cogvideox",
    "cosmos": "cosmos", "flux": "flux", "flux2": "flux2", "hidream": "hidream",
    "hunyuan_image": "hunyuan_image", "ideogram4": "ideogram4", "krea2": "krea2",
    "lens": "lens", "longcat_image": "longcat_image", "ltxv": "ltxv",
    "lumina2": "lumina2", "mochi": "mochi", "omnigen2": "omnigen2", "ovis": "ovis",
    "pid": "pixeldit", "pixart": "pixart", "pixeldit": "pixeldit", "pixl": "pixeldit",
    "qwen_image": "qwen_image", "sd3": "sd3", "stable_audio": "stable_audio",
    "stable_cascade": "stable_cascade", "stable_diffusion": "stable_diffusion", "wan": "wan",
}

# 需要附加 Flux 风格 guidance 的类型
GUIDANCE_CLIP_TYPES = {"flux", "flux2", "pid", "pixl", "boogu", "ideogram4", "pixeldit"}

def normalize_clip_type(clip_type):
    """把用户选择的 CLIP_类型 归一化为 ComfyUI 可识别的 CLIPType 名称。"""
    if not clip_type:
        return "stable_diffusion"
    key = clip_type.lower()
    return CLIP_TYPE_CANONICAL.get(key, "stable_diffusion")

# ========== GGUF 支持 ==========
# 复用 ComfyUI-GGUF（City96）的加载器，让本节点可直接选择 .gguf 扩散模型 / CLIP。
# 不复制其实现，而是按路径动态加载 ComfyUI-GGUF 的 nodes 模块，保证随上游同步更新。
_GGUF_NODES_CACHE = None

def _register_gguf_folders():
    """幂等地把 .gguf 文件登记到 unet_gguf / clip_gguf 目录键（与 ComfyUI-GGUF 一致），
    保证即便加载顺序不同也能在下拉框列出 .gguf 文件。"""
    def _add(key, targets):
        base = folder_paths.folder_names_and_paths.get(key, ([], {}))
        base = base[0] if isinstance(base[0], (list, set, tuple)) else []
        target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
        orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
        folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    _add("unet_gguf", ["diffusion_models", "unet"])
    _add("clip_gguf", ["text_encoders", "clip"])

_register_gguf_folders()

def _get_gguf_nodes():
    """按路径加载 ComfyUI-GGUF 的 nodes 模块并缓存。返回模块含
    UnetLoaderGGUF / CLIPLoaderGGUF / DualCLIPLoaderGGUF / GGMLOps /
    gguf_sd_loader / gguf_clip_loader / GGUFModelPatcher。"""
    global _GGUF_NODES_CACHE
    if _GGUF_NODES_CACHE is not None:
        return _GGUF_NODES_CACHE
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gguf_dir = os.path.join(custom_nodes_dir, "ComfyUI-GGUF")
    init_path = os.path.join(gguf_dir, "__init__.py")
    if not os.path.isfile(init_path):
        raise RuntimeError(
            "未找到 ComfyUI-GGUF 自定义节点（应在 custom_nodes/ComfyUI-GGUF）。"
            "加载 .gguf 模型需要先安装 ComfyUI-GGUF。"
        )
    pkg_name = "binyuan_gguf_loader"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_path, submodule_search_locations=[gguf_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    _GGUF_NODES_CACHE = sys.modules[pkg_name + ".nodes"]
    return _GGUF_NODES_CACHE

def load_gguf_diffusion_model(model_path):
    """通过 ComfyUI-GGUF 加载 .gguf 扩散模型，返回 ModelPatcher。
    GGUF 自带量化，不再额外套用 权重精度 的 dtype。"""
    gn = _get_gguf_nodes()
    ops = gn.GGMLOps()
    sd, extra = gn.gguf_sd_loader(model_path)
    kwargs = {}
    if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters:
        kwargs["metadata"] = extra.get("metadata", {})
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}, **kwargs
    )
    if model is None:
        raise RuntimeError(f"无法识别 GGUF 模型类型: {model_path}")
    model = gn.GGUFModelPatcher.clone(model)
    return model

def load_gguf_clip(clip_paths, clip_type_str):
    """通过 ComfyUI-GGUF 加载 .gguf CLIP（支持单/双编码器），返回 CLIP 对象。"""
    gn = _get_gguf_nodes()
    inst = gn.CLIPLoaderGGUF()
    clip_type = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    paths = list(clip_paths)
    clip_data = inst.load_data(paths)
    return inst.load_patcher(paths, clip_type, clip_data)

class BinyuanUltimateSamplerV9:
    @classmethod
    def INPUT_TYPES(s):
        def get_files(folder):
            return get_files_cached(folder)

        all_clip_files = []
        for folder in ["clip", "text_encoders", "CLIP", "clip_gguf"]:
            if folder in folder_paths.folder_names_and_paths:
                all_clip_files.extend(get_files_cached(folder))
        all_clip_files = sorted(list(set(all_clip_files)))

        # 扩散模型：合并 diffusion_models / unet / unet_gguf（含 .gguf），去重
        all_diffusion_files = sorted(list(set(
            get_files("diffusion_models") + get_files("unet") + get_files("unet_gguf")
        )))

        # V5.5 支持的全部类型（与 ComfyUI CLIPType 枚举对齐，大小写变体保留以匹配用户习惯）
        CLIP_TYPES = [
            "ACE", "boogu", "chroma", "cogvideox", "cosmos", "flux", "flux2",
            "hidream", "hunyuan_image", "ideogram4", "krea2", "lens", "longcat_image",
            "LTXV", "lumina2", "mochi", "omnigen2", "ovis", "pid", "PixArt",
            "pixeldit", "pixl", "qwen_image", "sd3", "SD3", "stable_audio",
            "stable_cascade", "stable_diffusion", "wan"
        ]

        PRESET_SIZES = [
            "自定义", "512x512", "512x768", "768x512", "768x768",
            "768x1024", "1024x768", "1024x1024", "1024x1280", "1280x1024",
            "1152x1152", "1152x1536", "1536x1152", "1216x832", "832x1216",
            "1280x1280", "1280x1920", "1920x1280", "1536x1536", "1536x2048",
            "2048x1536", "2048x2048", "1024x2048", "2048x1024", "4096x4096"
        ]

        SAMPLERS = comfy.samplers.KSampler.SAMPLERS
        SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

        return {
            "required": {
                "加载模式": (["整包Checkpoint", "分离式(Flux/SD3/扩散)"], {"default": "整包Checkpoint"}),
                "Checkpoint": (["None"] + get_files("checkpoints"), {"default": "None"}),
                "扩散模型": (["None"] + all_diffusion_files, {"default": "None"}),
                "CLIP_1": (["None"] + all_clip_files, {"default": "None"}),
                "CLIP_2": (["None"] + all_clip_files, {"default": "None"}),
                "CLIP_类型": (CLIP_TYPES, {"default": "flux"}),
                "VAE": (["baked_vae"] + get_files("vae"), {"default": "baked_vae"}),
                "权重精度": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "nvfp4", "pid", "pixl", "bf16", "fp16"], {"default": "default"}),
                "串联模式": (["使用自身模型", "继承上游模型", "自动检测"], {"default": "继承上游模型"}),
                "上游图像处理": (["重新VAE编码", "直接作为Latent", "自动选择"], {"default": "重新VAE编码"}),
                "Latent输入源": (["空Latent", "外部Latent优先", "上游图像优先", "上游图像拼接"], {"default": "上游图像优先"}),
                "正面提示词": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl"}),
                "负面提示词": ("STRING", {"multiline": True, "default": ""}),
                "尺寸助手": (PRESET_SIZES, {"default": "自定义"}),
                "宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "生成数量": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "步数": ("INT", {"default": 20, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "Flux引导": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "采样算法": (SAMPLERS, {"default": "euler"}),
                "调度器": (SCHEDULERS, {"default": "simple"}),
                "重绘强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lora_json": ("STRING", {"multiline": True, "default": "[]"}),
                "LORA_LIST": (["None"] + get_files("loras") + get_files("lora"),),
                "自动清理显存": ("BOOLEAN", {"default": False, "label": "生成后清理显存"}),
            },
            "optional": {
                "外部模型": ("MODEL",),
                "外部CLIP": ("CLIP",),
                "外部VAE": ("VAE",),
                "外部Latent": ("LATENT",),
                "外部正面条件": ("CONDITIONING",),
                "外部负面条件": ("CONDITIONING",),
                "上游图像_1": ("IMAGE",),
                "上游图像_2": ("IMAGE",),
                "上游图像_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("图像", "模型", "CLIP", "VAE", "Latent", "正面条件", "负面条件")
    FUNCTION = "run"
    CATEGORY = "Binyuan"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return None

    def get_image_size(self, image_tensor):
        if image_tensor is None:
            return 0, 0
        if len(image_tensor.shape) == 4:
            _, h, w, _ = image_tensor.shape
        else:
            h, w = image_tensor.shape[:2]
        return w, h

    def encode_image_to_latent(self, vae, image, width, height):
        try:
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
                image = F.interpolate(image, size=(height, width), mode='bilinear')
                image = image.permute(0, 2, 3, 1)
            return nodes.VAEEncode().encode(vae, image)[0]
        except Exception as e:
            print(f"[错误] 图像编码失败: {e}")
            return None

    def stitch_images_horizontal(self, images_list, target_h):
        """将多张上游图像横向拼接为单张画布（用于「上游图像拼接」模式）。"""
        parts = []
        for img in images_list:
            if img is None:
                continue
            t = img
            if t.dim() == 4:
                t = t[0]  # 取首张
            if t.dim() != 3:
                continue
            # 按目标高度等比缩放，宽度对齐到 8
            h0 = t.shape[0]
            if h0 <= 0:
                continue
            scale = float(target_h) / float(h0)
            new_w = max(8, int(round(t.shape[1] * scale) // 8 * 8))
            x = t.permute(2, 0, 1).unsqueeze(0).float()  # [1,C,H,W]
            x = F.interpolate(x, size=(int(target_h), new_w), mode='bilinear', align_corners=False)
            parts.append(x)
        if not parts:
            return None
        stitched = torch.cat(parts, dim=3)  # [1,C,H,W_total]
        out = stitched.permute(0, 2, 3, 1)  # [1,H,W_total,C]
        return out

    def get_latent_format(self, model):
        """安全读取模型 latent 格式（通道数 / 空间下采样比 / 维度）。"""
        try:
            return model.get_model_object("latent_format")
        except Exception:
            return None

    def empty_latent_for_model(self, model, width, height, batch):
        """按模型 latent 格式创建空 Latent（通道数与下采样比与模型匹配）。"""
        fmt = self.get_latent_format(model)
        channels = getattr(fmt, "latent_channels", 4) if fmt else 4
        ratio = getattr(fmt, "spacial_downscale_ratio", 8) if fmt else 8
        dims = getattr(fmt, "latent_dimensions", 2) if fmt else 2
        h = max(1, height // int(ratio))
        w = max(1, width // int(ratio))
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
        if dims == 3:
            # 视频模型：单帧
            latent = torch.zeros([batch, int(channels), 1, h, w], device=device, dtype=dtype)
        else:
            latent = torch.zeros([batch, int(channels), h, w], device=device, dtype=dtype)
        print(f"[信息] 空Latent: {latent.shape} (通道={channels}, 下采样比={ratio}, 维度={dims})")
        return {"samples": latent, "downscale_ratio_spacial": int(ratio)}

    def validate_vae(self, model, vae):
        """校验 VAE 与模型 latent 通道是否匹配，不匹配给出明确错误。"""
        if vae is None:
            return
        fmt = self.get_latent_format(model)
        model_ch = getattr(fmt, "latent_channels", None) if fmt else None
        vae_ch = getattr(vae, "latent_channels", None)
        if model_ch is not None and vae_ch is not None and int(model_ch) != int(vae_ch):
            raise RuntimeError(
                f"VAE 与模型不匹配：模型 latent 通道={model_ch}，当前 VAE 通道={vae_ch}。"
                f"请在 VAE 下拉框选择该模型对应的 VAE 文件。"
            )
        vae_ratio = getattr(vae, "downscale_ratio", None)
        print(f"[信息] VAE 校验通过 (VAE通道={vae_ch}, 下采样比={vae_ratio})")

    def get_model_name(self, model_obj):
        if model_obj is None:
            return "无"
        if hasattr(model_obj, 'model_config') and hasattr(model_obj.model_config, '_name'):
            return model_obj.model_config._name
        if hasattr(model_obj, 'model_type'):
            return str(model_obj.model_type)
        if hasattr(model_obj, '__class__'):
            return model_obj.__class__.__name__
        return "未知模型"

    def get_clip_name(self, clip_obj):
        if clip_obj is None:
            return "无"
        if hasattr(clip_obj, 'model_config') and hasattr(clip_obj.model_config, '_name'):
            return clip_obj.model_config._name
        if hasattr(clip_obj, '__class__'):
            return clip_obj.__class__.__name__
        return "未知CLIP"

    def get_vae_name(self, vae_obj):
        if vae_obj is None:
            return "无"
        if hasattr(vae_obj, '__class__'):
            return vae_obj.__class__.__name__
        return "未知VAE"

    def load_clip_internal(self, clip_1, clip_2, clip_type_canon):
        """加载 CLIP，自动按文件后缀在标准加载器与 GGUF 加载器间分流。
        任一 CLIP 文件为 .gguf 即整体走 GGUF 加载器（混用 scaled_fp8 与 gguf 不被支持）。"""
        if not clip_1 or clip_1 == "None":
            return None
        use_gguf = clip_1.lower().endswith(".gguf") or (
            clip_2 and clip_2 != "None" and clip_2.lower().endswith(".gguf")
        )
        names = [clip_1]
        if clip_2 and clip_2 != "None":
            names.append(clip_2)

        if use_gguf:
            paths = []
            for c in names:
                p = (folder_paths.get_full_path("clip", c) or
                     folder_paths.get_full_path("text_encoders", c) or
                     folder_paths.get_full_path("clip_gguf", c))
                if not p:
                    raise FileNotFoundError(f"找不到CLIP: {c}")
                paths.append(p)
            clip = load_gguf_clip(paths, clip_type_canon)
            print(f"[信息] 加载GGUF CLIP: {', '.join(names)} (类型: {clip_type_canon})")
            return clip

        if clip_2 and clip_2 != "None":
            loader = nodes.DualCLIPLoader()
            result = loader.load_clip(clip_1, clip_2, clip_type_canon)
        else:
            loader = nodes.CLIPLoader()
            result = loader.load_clip(clip_1, clip_type_canon)
        if result is not None and len(result) > 0:
            print(f"[信息] 加载CLIP: {clip_1} (类型: {clip_type_canon})")
            return result[0]
        return None

    def detect_model_type(self, lora_name):
        """检测 LoRA 类型，用于选择加载方式"""
        if lora_name is None:
            return "standard"
        lora_name_lower = lora_name.lower()
        if "klein" in lora_name_lower or "flux2" in lora_name_lower:
            return "klein"
        if "zimage" in lora_name_lower or "lumina2" in lora_name_lower or "turbo" in lora_name_lower:
            return "zimage"
        return "standard"

    def cleanup_vram(self):
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[信息] 显存已清理")

    def parse_lora_config(self, lora_json_str):
        """解析 LoRA 配置。兼容多种写法，保证任意内核下都能手动填写：
        - JSON 数组：[{"n":"x.safetensors","s":0.8,"sm":0.8,"sc":0.8,"e":true}, ...]
        - 单个 JSON 对象：{"n":"x.safetensors","s":0.8}
        - 简单逐行文本（每行一个 LoRA，# 或 // 开头为注释）：
            x.safetensors
            x.safetensors:0.8
            x.safetensors|0.8
        字段：n/name=文件名，s/strength=统一强度，sm=模型强度，sc=CLIP强度，e=是否启用。
        """
        if not lora_json_str:
            return []
        s = str(lora_json_str).strip()
        if not s or s in ("[]", "{}"):
            return []

        # 1) 优先按 JSON 解析
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except (json.JSONDecodeError, ValueError):
            pass

        # 2) 回退：逐行简单解析（即便前端 JS 全部失效也能手填）
        result = []
        for line in s.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            name = line
            strength = 1.0
            m = re.match(r"^(.+?)\s*[:|,]\s*(-?\d+(?:\.\d+)?)\s*$", line)
            if m:
                name = m.group(1).strip()
                strength = float(m.group(2))
            if name and name != "None":
                result.append({"n": name, "s": strength, "e": True})
        return result

    def load_loras(self, model, clip, lora_list):
        """统一的 LoRA 加载入口，支持多 LoRA 叠加。
        只依赖 ComfyUI 多年稳定的公开 API（comfy.utils.load_torch_file /
        comfy.sd.load_lora_for_models），不触碰易变的内部接口，保证跨内核可用。
        """
        loaded_count = 0
        for idx, lora in enumerate(lora_list):
            if not isinstance(lora, dict):
                continue
            if not lora.get("e", True):
                continue
            lora_name = lora.get("n") or lora.get("name")
            if not lora_name or lora_name == "None":
                continue

            # 强度：s 为统一强度；sm/sc 可分别覆盖模型/CLIP 强度
            strength = float(lora.get("s", lora.get("strength", 1.0)))
            sm = lora.get("sm")
            sc = lora.get("sc")
            sm = float(sm) if sm is not None else strength
            sc = float(sc) if sc is not None else strength

            # 查找文件（同时兼容 loras / lora 两个目录名）
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                lora_path = folder_paths.get_full_path("lora", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                print(f"[警告] LoRA文件不存在: {lora_name}")
                continue

            print(f"[信息] 加载 LoRA [{idx+1}]: {lora_name} (模型强度={sm}, CLIP强度={sc})")
            try:
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, sm, sc)
                loaded_count += 1
                print(f"[信息] LoRA 加载成功: {lora_name}")
            except Exception as e:
                print(f"[警告] LoRA 加载失败 {lora_name}: {e}")

        return model, clip, loaded_count

    def run(self, **kwargs):
        print(f"[信息] Binyuan采样器 V5.5")
        
        # ========== 获取并解析参数 ==========
        加载模式 = kwargs.get("加载模式", "整包Checkpoint")
        Checkpoint = kwargs.get("Checkpoint", "None")
        扩散模型 = kwargs.get("扩散模型", "None")
        CLIP_1 = kwargs.get("CLIP_1", "None")
        CLIP_2 = kwargs.get("CLIP_2", "None")
        CLIP_类型 = kwargs.get("CLIP_类型", "flux")
        CLIP_类型_规范 = normalize_clip_type(CLIP_类型)
        VAE = kwargs.get("VAE", "baked_vae")
        权重精度 = kwargs.get("权重精度", "default")
        串联模式 = kwargs.get("串联模式", "继承上游模型")
        上游图像处理 = kwargs.get("上游图像处理", "重新VAE编码")
        Latent输入源 = kwargs.get("Latent输入源", "上游图像优先")
        正面提示词 = kwargs.get("正面提示词", "masterpiece, best quality, 1girl")
        负面提示词 = kwargs.get("负面提示词", "")
        宽度 = int(kwargs.get("宽度", 1024))
        高度 = int(kwargs.get("高度", 1024))
        生成数量 = int(kwargs.get("生成数量", 1))
        seed = int(kwargs.get("seed", 0))
        步数 = int(kwargs.get("步数", 20))
        CFG = float(kwargs.get("CFG", 1.0))
        Flux引导 = float(kwargs.get("Flux引导", 3.5))
        采样算法 = kwargs.get("采样算法", "euler")
        调度器 = kwargs.get("调度器", "simple")
        重绘强度 = float(kwargs.get("重绘强度", 1.0))
        lora_json_str = kwargs.get("lora_json", "[]")
        自动清理显存 = kwargs.get("自动清理显存", False)
        
        # ========== 获取可选外部输入 ==========
        external_model = kwargs.get("外部模型")
        external_clip = kwargs.get("外部CLIP")
        external_vae = kwargs.get("外部VAE")
        external_latent = kwargs.get("外部Latent")
        external_positive = kwargs.get("外部正面条件")
        external_negative = kwargs.get("外部负面条件")
        
        upstream_image_1 = kwargs.get("上游图像_1")
        upstream_image_2 = kwargs.get("上游图像_2")
        upstream_image_3 = kwargs.get("上游图像_3")

        upstream_images = []
        if upstream_image_1 is not None:
            upstream_images.append(upstream_image_1)
        if upstream_image_2 is not None:
            upstream_images.append(upstream_image_2)
        if upstream_image_3 is not None:
            upstream_images.append(upstream_image_3)
        
        print(f"[信息] 上游图像: {len(upstream_images)} 张")
        print(f"[信息] 串联模式: {串联模式}")
        
        # ========== 打印外接状态 ==========
        print(f"[信息] 外部模型: {self.get_model_name(external_model)}")
        print(f"[信息] 外部CLIP: {self.get_clip_name(external_clip)}")
        print(f"[信息] 外部VAE: {self.get_vae_name(external_vae)}")
        print(f"[信息] 外部Latent: {'有' if external_latent is not None else '无'}")
        print(f"[信息] 外部正面条件: {'有' if external_positive is not None else '无'}")
        print(f"[信息] 外部负面条件: {'有' if external_negative is not None else '无'}")
        
        use_inherit = (串联模式 == "继承上游模型")
        
        model = None
        clip = None
        vae = None
        
        try:
            # ========== 解析模型权重精度配置 ==========
            model_options = {}
            te_model_options = {}
            
            if 权重精度 == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
                te_model_options["dtype"] = torch.float8_e4m3fn
            elif 权重精度 == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
                te_model_options["dtype"] = torch.float8_e4m3fn
            elif 权重精度 == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
                te_model_options["dtype"] = torch.float8_e5m2
            elif 权重精度 in ["nvfp4", "pid", "pixl"]:
                model_options["dtype"] = "nvfp4"
                te_model_options["dtype"] = "nvfp4"
            elif 权重精度 == "bf16":
                model_options["dtype"] = torch.bfloat16
                te_model_options["dtype"] = torch.bfloat16
            elif 权重精度 == "fp16":
                model_options["dtype"] = torch.float16
                te_model_options["dtype"] = torch.float16

            # ========== 模型加载 ==========
            if use_inherit and external_model is not None:
                print("[信息] 串联模式=继承上游模型，使用外接模型")
                model = external_model
                clip = external_clip if external_clip is not None else clip
                vae = external_vae if external_vae is not None else vae
                
                if clip is None and CLIP_1 and CLIP_1 != "None":
                    print(f"[信息] 外接缺少CLIP，从内部补充: {CLIP_1} (类型: {CLIP_类型} → {CLIP_类型_规范})")
                    clip = self.load_clip_internal(CLIP_1, CLIP_2, CLIP_类型_规范)
                
                if vae is None and VAE and VAE != "baked_vae" and VAE != "None":
                    print(f"[信息] 外接缺少VAE，从内部补充: {VAE}")
                    vae_path = folder_paths.get_full_path("vae", VAE)
                    if vae_path:
                        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            
            else:
                print("[信息] 串联模式=使用自身模型，使用内部配置")
                if 加载模式 == "整包Checkpoint":
                    if not Checkpoint or Checkpoint == "None":
                        raise ValueError("请选择Checkpoint")
                    checkpoint_path = folder_paths.get_full_path("checkpoints", Checkpoint)
                    if not checkpoint_path:
                        raise FileNotFoundError(f"找不到: {Checkpoint}")
                    
                    try:
                        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                            checkpoint_path, output_vae=True, output_clip=True,
                            model_options=model_options, te_model_options=te_model_options
                        )[:3]
                    except Exception as e:
                        print(f"[警告] 使用指定精度加载模型失败，降级回默认检测机制: {e}")
                        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                            checkpoint_path, output_vae=True, output_clip=True
                        )[:3]
                    print(f"[信息] 加载Checkpoint: {Checkpoint}")
                else:
                    if 扩散模型 and 扩散模型 != "None":
                        diffusion_path = (folder_paths.get_full_path("diffusion_models", 扩散模型) or
                                        folder_paths.get_full_path("unet", 扩散模型) or
                                        folder_paths.get_full_path("unet_gguf", 扩散模型))
                        if diffusion_path:
                            if 扩散模型.lower().endswith(".gguf"):
                                print(f"[信息] 加载GGUF扩散模型: {扩散模型}")
                                model = load_gguf_diffusion_model(diffusion_path)
                            else:
                                try:
                                    model = comfy.sd.load_diffusion_model(diffusion_path, model_options=model_options)
                                except Exception as e:
                                    print(f"[警告] 使用指定精度加载扩散模型失败，降级回默认检测机制: {e}")
                                    model = comfy.sd.load_diffusion_model(diffusion_path)
                                print(f"[信息] 加载扩散模型: {扩散模型}")
                        else:
                            raise FileNotFoundError(f"找不到扩散模型: {扩散模型}")
                    else:
                        raise ValueError("分离模式下必须选择扩散模型")
                    
                    if CLIP_1 and CLIP_1 != "None":
                        clip = self.load_clip_internal(CLIP_1, CLIP_2, CLIP_类型_规范)
                        if clip is None:
                            raise RuntimeError("CLIP加载失败")
                    else:
                        raise ValueError("分离模式下必须选择CLIP_1")
                    
                    if VAE and VAE != "baked_vae" and VAE != "None":
                        vae_path = folder_paths.get_full_path("vae", VAE)
                        if vae_path:
                            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
                            print(f"[信息] 加载VAE: {VAE}")
            
            if model is None:
                raise RuntimeError("模型加载失败")
            if clip is None:
                raise RuntimeError("CLIP加载失败")
            if vae is None:
                if hasattr(model, 'model') and hasattr(model.model, 'vae'):
                    vae = model.model.vae
                    print("[信息] 使用模型内置VAE")
                else:
                    raise RuntimeError(
                        "未加载 VAE。整包 Checkpoint 通常自带 VAE；分离式扩散模型不会自带 VAE，"
                        "必须在「VAE」下拉框选择该模型对应的 VAE 文件"
                        "（如 Flux/SD3 用 ae.safetensors，Wan/Qwen/Krea2 用 wan2.1 的 VAE 等）。"
                    )
            # 校验 VAE 与模型 latent 通道匹配，避免「能采样但解码失败」
            self.validate_vae(model, vae)
            
            # ========== LoRA 加载（多 LoRA 叠加，跨内核稳定） ==========
            lora_list = self.parse_lora_config(lora_json_str)
            if lora_list:
                print(f"[信息] 解析到 {len(lora_list)} 个LoRA")
                model, clip, loaded_count = self.load_loras(model, clip, lora_list)
                print(f"[信息] 共成功加载 {loaded_count} 个LoRA")
            else:
                print("[信息] 无LoRA配置")
            
            # ========== 条件编码 ==========
            def encode_prompt(text, clip_type_val, guidance_val):
                if not text or not text.strip():
                    text = " "
                tokens = clip.tokenize(text)
                # 必须用 return_dict=True：boogu/omnigen2 等模型的 transformer 需要 num_tokens，
                # 而 num_tokens 来自 attention_mask；只有 return_dict 才会把 attention_mask 等额外条件带出来。
                out = clip.encode_from_tokens(tokens, return_dict=True)
                cond = out.get("cond")
                pooled = out.get("pooled_output", None)
                cond_dict = {}
                if pooled is None:
                    if cond is not None and len(cond) > 0 and len(cond[0]) > 0:
                        pooled = torch.zeros_like(cond[0][0])
                    else:
                        pooled = torch.zeros((1, 2048))
                cond_dict["pooled_output"] = pooled
                # 复制 TE 返回的额外条件（attention_mask 等），供 omnigen2/boogu/qwen 等使用
                for k, v in out.items():
                    if k in ("cond", "pooled_output"):
                        continue
                    cond_dict[k] = v
                if clip_type_val in GUIDANCE_CLIP_TYPES:
                    cond_dict["guidance"] = guidance_val
                return [[cond, cond_dict]]

            if use_inherit and external_positive is not None:
                positive = external_positive
                print("[信息] 使用外部正面条件")
            else:
                positive = encode_prompt(正面提示词, CLIP_类型_规范, Flux引导)

            if use_inherit and external_negative is not None:
                negative = external_negative
                print("[信息] 使用外部负面条件")
            elif 负面提示词 and 负面提示词.strip():
                negative = encode_prompt(负面提示词, CLIP_类型_规范, Flux引导)
            else:
                # 空负面也要走 return_dict，保证 attention_mask/num_tokens 与正面一致
                negative = encode_prompt(" ", CLIP_类型_规范, Flux引导)
            
            # ========== Latent ==========
            latent = None
            if use_inherit and external_latent is not None:
                latent = external_latent
                print("[信息] 使用外部Latent")
            elif Latent输入源 == "外部Latent优先" and external_latent is not None:
                latent = external_latent
                print("[信息] 使用外部Latent（优先）")
            elif Latent输入源 == "空Latent":
                latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                print(f"[信息] 使用空Latent: {宽度}x{高度}")
            elif Latent输入源 == "上游图像拼接" and len(upstream_images) >= 1:
                # 将多张上游图像横向拼接为一张画布后编码为 Latent
                w0, h0 = self.get_image_size(upstream_images[0])
                target_h = h0 if h0 and h0 > 0 else 高度
                stitched = self.stitch_images_horizontal(upstream_images, target_h)
                if stitched is not None:
                    sh, sw = stitched.shape[1], stitched.shape[2]
                    print(f"[信息] 拼接上游图像: {len(upstream_images)} 张 → 画布 {sw}x{sh}")
                    latent = self.encode_image_to_latent(vae, stitched, sw, sh)
                if latent is None:
                    w, h = self.get_image_size(upstream_images[0])
                    latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                if latent is None:
                    latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    print(f"[信息] 拼接编码失败，使用空Latent")
            elif len(upstream_images) >= 1:
                # 上游图像优先 / 自动选择
                w, h = self.get_image_size(upstream_images[0])
                print(f"[信息] 编码上游图像: {w}x{h}")
                latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                if latent is None:
                    latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    print(f"[信息] 编码失败，使用空Latent")
            else:
                latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                print(f"[信息] 使用空Latent: {宽度}x{高度}")
            
            if latent is None:
                raise RuntimeError("Latent 生成失败")
            
            # ========== 采样 ==========
            samples = nodes.common_ksampler(
                model, seed, 步数, CFG, 采样算法, 调度器,
                positive, negative, latent, denoise=重绘强度
            )
            
            images = vae.decode(samples[0]["samples"])
            # 视频 VAE（如 Wan/Qwen/Krea2 用的 Wan21 VAE，latent_dim=3）解码后是 5D [B,T,H,W,C]，
            # 需要把时间维合并进 batch，得到标准 4D IMAGE [B,H,W,C]，否则 SaveImage 会报
            # "Cannot handle this data type"。与 ComfyUI 官方 VAEDecode 处理一致。
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            elif len(images.shape) == 3:
                images = images.unsqueeze(0)
            print(f"[信息] 生成完成，图像数量: {images.shape[0]}")
            
            if 自动清理显存:
                self.cleanup_vram()
            
            return (images, model, clip, vae, samples[0], positive, negative)
            
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()
            if 自动清理显存:
                self.cleanup_vram()
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, None, None, None, None, None, None)

NODE_CLASS_MAPPINGS = {"BinyuanUltimateSampler": BinyuanUltimateSamplerV9}
NODE_DISPLAY_NAME_MAPPINGS = {"BinyuanUltimateSampler": "🛡️ Binyuan采样器 V5.5"}
WEB_DIRECTORY = "js"