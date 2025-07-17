# ComfyUI_SZtools
1. This project is the comfyui implementation of ComfyUI_SZtools, a labeling and naming tool developed for Kontext's local training package T2ITrainer.
2. Lrzjason/T2iTraining Address: https://github.com/lrzjason/T2ITrainer
The naming convention for the T2ITrainer training set is: xxx_ R.png, xxx_ T. png, xxx_ R.txt
Batch naming of images as comfyui0001_T.png and comfyui0001_T.txt is achieved by saving nodes through the T-mangosteen Kontext path in ComfyUI_SZtools
Batch naming of images as comfyui0001_R.png (can/cannot save reverse text) is achieved by saving the R-mangosteen Kontext path node in ComfyUI_SZtools
Thus, when T2ITrainer trains Kontext LoRA locally, it can quickly achieve the task of marking the training set (requiring the combination of other reverse inference prompt word nodes)
4. Installation method:
Install through Git
git clone  https://github.com/comfyanonymous/ComfyUI.git
Or download the installation package
6. Installation environment dependencies:
python.exe -m pip install -r D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_SZtools\requirements.txt
8. ComfyUI_SZtools also includes other nodes.
Thank you to lrzjason/T2iTraining for their work.
10. The homepage of AIGI's Bilibili for Teacher Mangosteen: https://space.bilibili.com/158424637?spm_id_from=333.1007.0.0

# ComfyUI_SZtools
1.本项目是ComfyUI_SZtools的comfyui实现，为Kontext的本地训练包T2ITrainer而开发的打标命名工具。  

lrzjason/T2ITrainer地址：https://github.com/lrzjason/T2ITrainer  

2.T2ITrainer训练集的命名规则为：xxx_R.png,xxx_T.png,xxx_R.txt  

     通过ComfyUI_SZtools中的T-山竹Kontext路径保存节点实现对图像批量命名为comfyui0001_T.png和comfyui0001_T.txt
     通过ComfyUI_SZtools中的R-山竹Kontext路径保存节点实现对图像批量命名为comfyui0001_R.png(可/不可保存反推文本）
     从而实现T2ITrainer本地训练Kontext-LoRA时，快速实现训练集打标重命名的任务（需要结合其他反推提示词节点）
3.安装方法：  

通过git安装
git clone https://github.com/comfyanonymous/ComfyUI.git  

或下载安装包  

4.安装环境依赖：  

python.exe -m pip install -r D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_SZtools\requirements.txt  

5.ComfyUI_SZtools也包含其他节点。  

6.感谢lrzjason/T2ITrainer所做的工作。  

7.山竹老师AIGI的B站主页：https://space.bilibili.com/158424637?spm_id_from=333.1007.0.0  

