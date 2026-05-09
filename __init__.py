import os
import torch
import json
import folder_paths
import nodes
import comfy.utils
import comfy.sd
import comfy.samplers
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time

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

class BinyuanUltimateSamplerV9:
    @classmethod
    def INPUT_TYPES(s):
        def get_files(folder):
            return get_files_cached(folder)

        all_clip_files = []
        for folder in ["clip", "text_encoders", "CLIP"]:
            if folder in folder_paths.folder_names_and_paths:
                all_clip_files.extend(get_files_cached(folder))
        all_clip_files = sorted(list(set(all_clip_files)))

        CLIP_TYPES = [
            "flux2", "flux", "stable_diffusion", "StableCascade", "SD3",
            "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2",
            "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image",
            "hunyuan_image", "ovis", "longcat_image"
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
                "扩散模型": (["None"] + get_files("diffusion_models") + get_files("unet"), {"default": "None"}),
                "CLIP_1": (["None"] + all_clip_files, {"default": "None"}),
                "CLIP_2": (["None"] + all_clip_files, {"default": "None"}),
                "CLIP_类型": (CLIP_TYPES, {"default": "flux"}),
                "VAE": (["baked_vae"] + get_files("vae"), {"default": "baked_vae"}),
                "权重精度": (["default", "fp8_e4m3fn", "bf16", "fp16"], {"default": "default"}),
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
                "lora_json": ("STRING", {"default": "[]"}),
                "LORA_LIST": (["None"] + get_files("loras") + get_files("lora"),),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("图像", "模型", "CLIP", "VAE", "Latent", "正面条件", "负面条件")
    FUNCTION = "run"
    CATEGORY = "Binyuan"

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

    def run(self, 加载模式, Checkpoint, 扩散模型, CLIP_1, CLIP_2, CLIP_类型, VAE, 权重精度,
            正面提示词, 负面提示词, 串联模式="继承上游模型", 上游图像处理="重新VAE编码",
            Latent输入源="上游图像优先", 宽度=1024, 高度=1024, 生成数量=1, seed=0,
            步数=20, CFG=1.0, Flux引导=3.5, 采样算法="euler", 调度器="simple",
            重绘强度=1.0, **kwargs):
        
        print(f"[信息] Binyuan采样器 V4.2")
        
        # ========== 获取外部输入 ==========
        external_model = kwargs.get("外部模型")
        external_clip = kwargs.get("外部CLIP")
        external_vae = kwargs.get("外部VAE")
        external_latent = kwargs.get("外部Latent")
        external_positive = kwargs.get("外部正面条件")
        external_negative = kwargs.get("外部负面条件")
        
        upstream_image_1 = kwargs.get("上游图像_1")
        upstream_image_2 = kwargs.get("上游图像_2")
        
        upstream_images = []
        if upstream_image_1 is not None:
            upstream_images.append(upstream_image_1)
        if upstream_image_2 is not None:
            upstream_images.append(upstream_image_2)
        
        print(f"[信息] 上游图像: {len(upstream_images)} 张")
        print(f"[信息] 串联模式: {串联模式}")
        
        # ========== 打印外接状态（打印名称） ==========
        print(f"[信息] 外部模型: {self.get_model_name(external_model)}")
        print(f"[信息] 外部CLIP: {self.get_clip_name(external_clip)}")
        print(f"[信息] 外部VAE: {self.get_vae_name(external_vae)}")
        print(f"[信息] 外部Latent: {'有' if external_latent is not None else '无'}")
        print(f"[信息] 外部正面条件: {'有' if external_positive is not None else '无'}")
        print(f"[信息] 外部负面条件: {'有' if external_negative is not None else '无'}")
        
        # ========== 核心逻辑：只有串联模式为"继承上游模型"时，才使用外接 ==========
        use_inherit = (串联模式 == "继承上游模型")
        
        model = None
        clip = None
        vae = None
        
        try:
            # ========== 模型加载 ==========
            if use_inherit and external_model is not None:
                print("[信息] 串联模式=继承上游模型，使用外接模型")
                model = external_model
                clip = external_clip if external_clip is not None else clip
                vae = external_vae if external_vae is not None else vae
                
                if clip is None and CLIP_1 and CLIP_1 != "None":
                    print(f"[信息] 外接缺少CLIP，从内部补充: {CLIP_1}")
                    if CLIP_2 and CLIP_2 != "None":
                        loader = nodes.DualCLIPLoader()
                        result = loader.load_clip(CLIP_1, CLIP_2, CLIP_类型)
                        if result is not None and len(result) > 0:
                            clip = result[0]
                    else:
                        loader = nodes.CLIPLoader()
                        result = loader.load_clip(CLIP_1, CLIP_类型)
                        if result is not None and len(result) > 0:
                            clip = result[0]
                
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
                    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                        checkpoint_path, output_vae=True, output_clip=True
                    )[:3]
                    print(f"[信息] 加载Checkpoint: {Checkpoint}")
                else:
                    if 扩散模型 and 扩散模型 != "None":
                        diffusion_path = (folder_paths.get_full_path("diffusion_models", 扩散模型) or
                                        folder_paths.get_full_path("unet", 扩散模型))
                        if diffusion_path:
                            model = comfy.sd.load_diffusion_model(diffusion_path)
                            print(f"[信息] 加载扩散模型: {扩散模型}")
                        else:
                            raise FileNotFoundError(f"找不到扩散模型: {扩散模型}")
                    else:
                        raise ValueError("分离模式下必须选择扩散模型")
                    
                    if CLIP_1 and CLIP_1 != "None":
                        if CLIP_2 and CLIP_2 != "None":
                            loader = nodes.DualCLIPLoader()
                            result = loader.load_clip(CLIP_1, CLIP_2, CLIP_类型)
                            if result is not None and len(result) > 0:
                                clip = result[0]
                        else:
                            loader = nodes.CLIPLoader()
                            result = loader.load_clip(CLIP_1, CLIP_类型)
                            if result is not None and len(result) > 0:
                                clip = result[0]
                        print(f"[信息] 加载CLIP: {CLIP_1}")
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
                    raise RuntimeError("VAE加载失败")
            
            # ========== LoRA 加载 ==========
            lora_json_str = kwargs.get("lora_json", "[]")
            if lora_json_str and lora_json_str not in ["", "[]", "{}"]:
                try:
                    lora_list = json.loads(lora_json_str)
                    for lora in lora_list:
                        if lora.get("e") and lora.get("n") and lora.get("n") != "None":
                            lora_name = lora["n"]
                            lora_strength = lora.get("s", 1.0)
                            lora_path = folder_paths.get_full_path("loras", lora_name)
                            if not lora_path:
                                lora_path = folder_paths.get_full_path("lora", lora_name)
                            if lora_path and os.path.exists(lora_path):
                                lora_sd = comfy.utils.load_torch_file(lora_path)
                                model, clip = comfy.sd.load_lora_for_models(
                                    model, clip, lora_sd, lora_strength, lora_strength
                                )
                                print(f"[信息] LoRA加载成功: {lora_name} 强度: {lora_strength}")
                            else:
                                print(f"[警告] LoRA文件不存在: {lora_name}")
                except Exception as e:
                    print(f"[警告] LoRA加载失败: {e}")
            
            # ========== 条件编码 ==========
            def encode_prompt(text, clip_type_val, guidance_val):
                if not text or not text.strip():
                    text = " "
                tokens = clip.tokenize(text)
                output = clip.encode_from_tokens(tokens, return_pooled=True)
                if isinstance(output, tuple) and len(output) == 2:
                    cond, pooled = output
                else:
                    cond = output
                    pooled = None
                if pooled is None:
                    if cond is not None and len(cond) > 0 and len(cond[0]) > 0:
                        pooled = torch.zeros_like(cond[0][0])
                    else:
                        pooled = torch.zeros((1, 2048))
                cond_dict = {"pooled_output": pooled}
                if clip_type_val in ["flux", "flux2"]:
                    cond_dict["guidance"] = guidance_val
                return [[cond, cond_dict]]
            
            if use_inherit and external_positive is not None:
                positive = external_positive
                print("[信息] 使用外部正面条件")
            else:
                positive = encode_prompt(正面提示词, CLIP_类型, Flux引导)
            
            if use_inherit and external_negative is not None:
                negative = external_negative
                print("[信息] 使用外部负面条件")
            elif 负面提示词 and 负面提示词.strip():
                negative = encode_prompt(负面提示词, CLIP_类型, Flux引导)
            else:
                empty_text = " "
                tokens = clip.tokenize(empty_text)
                output = clip.encode_from_tokens(tokens, return_pooled=True)
                if isinstance(output, tuple) and len(output) == 2:
                    empty_cond, empty_pooled = output
                else:
                    empty_cond = output
                    empty_pooled = None
                if empty_pooled is None:
                    empty_pooled = torch.zeros_like(empty_cond[0][0]) if empty_cond is not None else torch.zeros((1, 2048))
                negative = [[empty_cond, {"pooled_output": empty_pooled}]]
            
            # ========== Latent ==========
            latent = None
            if use_inherit and external_latent is not None:
                latent = external_latent
                print("[信息] 使用外部Latent")
            elif Latent输入源 == "空Latent":
                latent = nodes.EmptyLatentImage().generate(宽度, 高度, 生成数量)[0]
                print(f"[信息] 使用空Latent: {宽度}x{高度}")
            elif len(upstream_images) >= 1:
                w, h = self.get_image_size(upstream_images[0])
                print(f"[信息] 编码上游图像: {w}x{h}")
                latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                if latent is None:
                    latent = nodes.EmptyLatentImage().generate(宽度, 高度, 生成数量)[0]
                    print(f"[信息] 编码失败，使用空Latent")
            else:
                latent = nodes.EmptyLatentImage().generate(宽度, 高度, 生成数量)[0]
                print(f"[信息] 使用空Latent: {宽度}x{高度}")
            
            if latent is None:
                raise RuntimeError("Latent 生成失败")
            
            # ========== 采样 ==========
            samples = nodes.common_ksampler(
                model, seed, 步数, CFG, 采样算法, 调度器,
                positive, negative, latent, denoise=重绘强度
            )
            
            images = vae.decode(samples[0]["samples"])
            print(f"[信息] 生成完成，图像数量: {images.shape[0]}")
            
            return (images, model, clip, vae, samples[0], positive, negative)
            
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, None, None, None, None, None, None)

NODE_CLASS_MAPPINGS = {"BinyuanUltimateSampler": BinyuanUltimateSamplerV9}
NODE_DISPLAY_NAME_MAPPINGS = {"BinyuanUltimateSampler": "🛡️ Binyuan采样器 V4.2"}
WEB_DIRECTORY = "js"