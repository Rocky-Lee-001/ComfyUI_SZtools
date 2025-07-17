import torch
import numpy as np
import os
import folder_paths
import imageio
import tifffile

class ShanZhuDepthVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "pixels": ("IMAGE", ),
                "vae": ("VAE", ),
                "smooth_factor": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "山竹VAE深度编码器"

    def encode(self, vae, pixels, smooth_factor):
        # 确保所有计算在VAE设备上执行
        device = vae.device
        pixels = pixels.to(device)
        
        # 获取原始潜在向量（自动匹配VAE设备）
        orig_latent = vae.encode(pixels[:,:,:,:3])
        if isinstance(orig_latent, dict):
            orig_latent = orig_latent["samples"]
        
        # 直接在原始潜在向量的设备上生成噪声
        noise = torch.randn_like(orig_latent) * 0.5  # 自动继承设备信息
        
        # 混合操作（确保所有张量在同一设备）
        blended = orig_latent * (1 - smooth_factor) + noise * smooth_factor
        
        return ({"samples": blended}, )
        
class ShanzhuDepthVAEDecoder:
    """深度增强VAE解码器，强制32位精度输出的潜在变量解码，噪声范围0.001~0.1"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "待解码的潜在空间张量"}),
                "vae": ("VAE", {"tooltip": "VAE解码模型"})
            },
            "optional": {
                "noise_strength": ("FLOAT", {
                    "default": 0.001,
                    "min": 0.000,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "噪声强度 (0.000-0.1)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("decoded_image",)
    FUNCTION = "advanced_decode"
    CATEGORY = "山竹VAE深度解码"

    def advanced_decode(self, vae, samples, noise_strength=0.000):
        latent = samples["samples"]
        decoded = vae.decode(latent).float()
        
        if noise_strength > 0:
            noise = torch.randn_like(decoded) * noise_strength
            decoded = torch.clamp(decoded + noise, 0.0, 1.0)
            
        return (decoded,)

class ShanzhuVAEDecodeSmooth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image in 16-bit TIFF format.",)
    FUNCTION = "decode_smooth"

    CATEGORY = "山竹VAE平滑解码器"
    DESCRIPTION = "Decodes latent images back into pixel space images in 16-bit TIFF format to reduce banding artifacts."

    def decode_smooth(self, vae, samples):
        # 使用 VAE 解码潜在表示
        images = vae.decode(samples["samples"])
        
        # 如果图像有 5 个维度，则重塑为 4 维
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        
        # 将 PyTorch 张量转换为 NumPy 数组
        images_np = images.cpu().numpy()
        
        # 将图像数据从 [0, 1] 范围缩放到 [0, 65535]，转换为 16 位无符号整数
        images_np = (images_np * 65535).astype(np.uint16)
        
        # 转置为 (B, H, W, C) 格式以便保存
        images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        # 将批次中的每个图像保存为 16 位 TIFF 文件
        for i, img in enumerate(images_np):
            filename = f"decoded_image_{i}.tif"
            tifffile.imwrite(filename, img)
        
        # 返回原始图像张量以兼容 ComfyUI 流程
        return (images, )
        
class ShanzhuTifSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "山竹TIF保存"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
        results = []
        for image in images:
            # 保存16位TIF文件
            i = 65535. * image.cpu().numpy()
            img = np.clip(i, 0, 65535).astype(np.uint16)
            file = f"{filename}_{counter:05}_.tif"
            imageio.imwrite(os.path.join(full_output_folder, file), img)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class ShanzhuLujingTIFSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "文件名前缀": ("STRING", {"default": "ComfyUI"}),
                "保存路径": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "山竹TIF路径保存"

    def save_images(self, images, 文件名前缀="ComfyUI", 保存路径="", prompt=None, extra_pnginfo=None):
        文件名前缀 += self.prefix_append
        
        if 保存路径.strip():
            output_dir = 保存路径
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.output_dir

        results = []
        saved_paths = []
        
        # 初始化计数器
        existing_files = []
        for img in images:
            # 获取每张图片的原始尺寸
            h, w = img.shape[0], img.shape[1]
            
            # 为每张图片单独生成保存路径
            full_output_folder, base_filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                文件名前缀, output_dir, w, h
            )
            
            # 生成唯一文件名（包含尺寸信息）
            file = f"{base_filename}_{counter:05}_{w}x{h}_.tif"
            img_path = os.path.join(full_output_folder, file)
            
            # 确保不覆盖已有文件
            while os.path.exists(img_path):
                counter += 1
                file = f"{base_filename}_{counter:05}_{w}x{h}_.tif"
                img_path = os.path.join(full_output_folder, file)
            
            # 转换并保存16位灰度TIF
            i = 65535.0 * img.cpu().numpy()
            
            # RGB转灰度处理
            if i.ndim == 3 and i.shape[2] == 3:
                # 使用标准RGB转灰度系数
                gray = np.dot(i[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                # 处理单通道或非常规情况
                gray = i.squeeze()
            
            # 确保二维数组并转换类型
            gray = np.clip(gray, 0, 65535).astype(np.uint16)
            imageio.imwrite(img_path, gray)
            
            # 记录保存信息
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            saved_paths.append(img_path)

        return {
            "ui": {
                "images": results,
                "保存路径": "\n".join(saved_paths)
            }
        }

        
NODE_CLASS_MAPPINGS = {
    "ShanZhuDepthVAEEncode": ShanZhuDepthVAEEncode,
    "ShanzhuDepthVAEDecoder": ShanzhuDepthVAEDecoder,
    "ShanzhuVAEDecodeSmooth": ShanzhuVAEDecodeSmooth,
    "ShanzhuTifSaver": ShanzhuTifSaver,
    "ShanzhuLujingTIFSaver": ShanzhuLujingTIFSaver 
       
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShanZhuDepthVAEEncode": "山竹VAE深度编码器",
    "ShanzhuDepthVAEDecoder": "山竹VAE深度解码器 ",
    "ShanzhuVAEDecodeSmooth": "山竹VAE平滑解码器",
    "ShanzhuTifSaver": "山竹TIF保存",
    "ShanzhuLujingTIFSaver": "山竹TIF路径保存"
   
}