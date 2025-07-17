import os
import shutil
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import datetime
import torch
import numpy as np
import folder_paths
import random
import string
from PIL import Image, ImageOps

# --- 辅助函数替代原来的 imagefunc 模块 ---

def log(message, message_type='info'):
    prefix = {
        'info': '[INFO]',
        'warning': '[WARN]',
        'error': '[ERROR]'
    }.get(message_type.lower(), '[INFO]')
    print(f"{prefix} {message}")

def generate_random_name(prefix='', suffix='', length=8):
    letters = string.ascii_letters + string.digits
    rand_str = ''.join(random.choice(letters) for _ in range(length))
    return f"{prefix}{rand_str}{suffix}"

def remove_empty_lines(text):
    return '\n'.join(line for line in text.split('\n') if line.strip())

# --- 原来的类定义 ---

class TSZImageTaggerSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.NODE_NAME = 'ImageTaggerSave'
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tag_text": ("STRING", {"default": "", "forceInput": True}),
                "custom_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "comfyui"}),
                "format": (["png", "jpg"],),
                "quality": ("INT", {"default": 100, "min": 80, "max": 100, "step": 1}),
                "preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "image_tagger_save"
    OUTPUT_NODE = True
    CATEGORY = 'T-山竹Kontext路径保存'

    def image_tagger_save(self, image, tag_text, custom_path, filename_prefix, format, quality, preview,
                           prompt=None, extra_pnginfo=None):
        folder = custom_path.strip() or self.output_dir
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            log(f"{self.NODE_NAME} -> 无法创建路径 '{folder}': {e}", message_type='error')
            return {}

        batch_count = image.shape[0]
        log(f"{self.NODE_NAME} -> Processing {batch_count} images...")

        results = []
        temp_dir = generate_random_name('_preview_', '_temp', 16)
        os.makedirs(os.path.join(folder_paths.get_temp_directory(), temp_dir), exist_ok=True)

        for img_tensor in image:
            arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            self.counter = (self.counter + 1) if (self.counter < 9999) else 1
            index_str = f"{self.counter:04d}"

            # 修改文件名格式：去掉下划线
            base = f"{filename_prefix}{index_str}_T"
            img_path = os.path.join(folder, f"{base}.{format}")

            if format == 'png':
                compress = max(0, min(9, (100 - quality) // 10))
                img.save(img_path, compress_level=compress)
            else:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(img_path, quality=quality)

            tag_path = os.path.join(folder, f"{base}.txt")
            with open(tag_path, 'w', encoding='utf-8') as f:
                f.write(remove_empty_lines(tag_text))

            log(f"{self.NODE_NAME} -> Saved {base}.{format} and {base}.txt (counter={self.counter})")
            results.append(img)

        ui_images = []
        if preview and results:
            preview_img = results[-1]
            preview_name = f"{generate_random_name('preview_', '', 16)}.png"
            preview_folder = os.path.join(folder_paths.get_temp_directory(), temp_dir)
            preview_path = os.path.join(preview_folder, preview_name)
            try:
                preview_img.save(preview_path)
                ui_images.append({"filename": preview_name, "subfolder": temp_dir, "type": 'temp'})
            except Exception as e:
                log(f"{self.NODE_NAME} -> 预览生成失败: {e}", message_type='warning')

        return {"ui": {"images": ui_images}}

class RSZImageTaggerSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.NODE_NAME = 'ImageTaggerSave'
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tag_text": ("STRING", {"default": ""}),
                "custom_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "comfyui"}),
                "format": (["png", "jpg"],),
                "quality": ("INT", {"default": 100, "min": 80, "max": 100, "step": 1}),
                "preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "image_tagger_save"
    OUTPUT_NODE = True
    CATEGORY = 'R-山竹Kontext路径保存'

    def image_tagger_save(self, image, tag_text, custom_path, filename_prefix, format, quality, preview,
                           prompt=None, extra_pnginfo=None):
        folder = custom_path.strip() or self.output_dir
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            log(f"{self.NODE_NAME} -> 无法创建路径 '{folder}': {e}", message_type='error')
            return {}

        batch_count = image.shape[0]
        log(f"{self.NODE_NAME} -> Processing {batch_count} images...")

        results = []
        temp_dir = generate_random_name('_preview_', '_temp', 16)
        os.makedirs(os.path.join(folder_paths.get_temp_directory(), temp_dir), exist_ok=True)

        for img_tensor in image:
            arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            self.counter = (self.counter + 1) if (self.counter < 9999) else 1
            index_str = f"{self.counter:04d}"

            # 修改文件名格式：去掉下划线
            base = f"{filename_prefix}{index_str}_R"
            img_path = os.path.join(folder, f"{base}.{format}")

            if format == 'png':
                compress = max(0, min(9, (100 - quality) // 10))
                img.save(img_path, compress_level=compress)
            else:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(img_path, quality=quality)
                
            # ✅ 只在 tag_text 非空时保存 .txt 文件
            if tag_text.strip():
                tag_path = os.path.join(folder, f"{base}.txt")
                with open(tag_path, 'w', encoding='utf-8') as f:
                    f.write(remove_empty_lines(tag_text))
            else:
                log(f"{self.NODE_NAME} -> 未保存文本文件（tag_text为空）", message_type='info')

            log(f"{self.NODE_NAME} -> Saved {base}.{format} and {base}.txt (counter={self.counter})")
            results.append(img)

        ui_images = []
        if preview and results:
            preview_img = results[-1]
            preview_name = f"{generate_random_name('preview_', '', 16)}.png"
            preview_folder = os.path.join(folder_paths.get_temp_directory(), temp_dir)
            preview_path = os.path.join(preview_folder, preview_name)
            try:
                preview_img.save(preview_path)
                ui_images.append({"filename": preview_name, "subfolder": temp_dir, "type": 'temp'})
            except Exception as e:
                log(f"{self.NODE_NAME} -> 预览生成失败: {e}", message_type='warning')

        return {"ui": {"images": ui_images}}
        
class ShanZhuTextSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = "_text"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "texts": ("STRING", {"multiline": True}),
                "文件名前缀": ("STRING", {"default": "ComfyUI"}),
                "保存路径": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_texts"
    OUTPUT_NODE = True
    CATEGORY = "山竹文本保存"

    def save_texts(self, images, texts, 文件名前缀="ComfyUI", 保存路径="", prompt=None, extra_pnginfo=None):
        文件名前缀 += self.prefix_append
        
        if 保存路径.strip():
            output_dir = 保存路径
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.output_dir

        text_list = [t.strip() for t in texts.split('\n')]
        if len(images) != len(text_list):
            raise ValueError("错误：图像数量与文本数量不一致")

        saved_paths = []
        
        for index, (img, text) in enumerate(zip(images, text_list)):
            w = img.shape[2]
            h = img.shape[1]
            
            base_name = f"{文件名前缀}_{index:05}"
            size_suffix = f"_{w}x{h}"
            file_name = f"{base_name}{size_suffix}.txt"
            
            txt_path = os.path.join(output_dir, file_name)
            
            counter = 1
            while os.path.exists(txt_path):
                file_name = f"{base_name}_{counter:05}{size_suffix}.txt"
                txt_path = os.path.join(output_dir, file_name)
                counter += 1
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            saved_paths.append(txt_path)

        return {
            "ui": {
                "texts": saved_paths,
                "保存路径": "\n".join(saved_paths)
            }
        }

sort_methods = ["name", "date", "size"]

# 示例排序函数（需根据实际需求完善）
def sort_by(file_list, directory, method):
    if method == "name":
        return sorted(file_list)
    elif method == "date":
        return sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    elif method == "size":
        return sorted(file_list, key=lambda x: os.path.getsize(os.path.join(directory, x)))
    else:
        return file_list  # 默认不排序

class ShanZhuLoadImagesFromDirList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 100, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE PATH")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "山竹导入图像路径列表"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs.items()))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.jxl']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sort_by(dir_files, directory, sort_method)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = image_load_cap > 0
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            file_paths.append(str(image_path))
            image_count += 1

        return (images, masks, file_paths)
        
class ShanZhuTextJoin:

    def __init__(self):
        self.NODE_NAME = 'ShanZhuTextJoin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"default": "", "multiline": False,"forceInput":False}),

            },
            "optional": {
                "text_2": ("STRING", {"default": "", "multiline": False,"forceInput":False}),
                "text_3": ("STRING", {"default": "", "multiline": False,"forceInput":False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '山竹文本合并'

    def text_join(self, text_1, text_2="", text_3=""):

        texts = []
        if text_1 != "":
            texts.append(text_1)
        if text_2 != "":
            texts.append(text_2)
        if text_3 != "":
            texts.append(text_3)
        if len(texts) > 0:
            combined_text = ', '.join(texts)
            return (combined_text.encode('unicode-escape').decode('unicode-escape'),)
        else:
            return ('',)        
        
NODE_CLASS_MAPPINGS = {
    "TSZImageTaggerSave": TSZImageTaggerSave,
    "RSZImageTaggerSave":RSZImageTaggerSave,
    "ShanZhuTextSaver": ShanZhuTextSaver,
    "ShanZhuLoadImagesFromDirList":ShanZhuLoadImagesFromDirList,
    "ShanZhuTextJoin":ShanZhuTextJoin
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSZImageTaggerSave": "T-山竹Kontext路径保存",
    "RSZImageTaggerSave":"R-山竹kontext路径保存",
    "ShanZhuTextSaver": "山竹文本路径保存",
    "ShanZhuLoadImagesFromDirList":"山竹导入图像路径列表",
    "ShanZhuTextJoin":"山竹文本合并"
}
