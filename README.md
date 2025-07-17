# ComfyUI_SZtools
1. This project is the comfyui implementation of ComfyUI_SZtools, a labeling and naming tool developed for Kontext's local training package T2ITrainer. Lrzjason/T2iTraining Address: https://github.com/lrzjason/T2ITrainer
The naming convention for the T2ITrainer training set is: xxx_ R.png, xxx_ T. png, xxx_ R.txt
Batch naming of images as comfyui0001_T.png and comfyui0001_T.txt is achieved by saving nodes through the T-mangosteen Kontext path in ComfyUI_SZtools
Batch naming of images as comfyui0001_R.png (can/cannot save reverse text) is achieved by saving the R-mangosteen Kontext path node in ComfyUI_SZtools
Thus, when T2ITrainer trains Kontext LoRA locally, it can quickly achieve the task of marking the training set (requiring the combination of other reverse inference prompt word nodes)
3. Installation method:
Install through Git
git clone  https://github.com/comfyanonymous/ComfyUI.git
Or download the installation package
4. Installation environment dependencies:
python.exe -m pip install -r D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_SZtools\requirements.txt
5. ComfyUI_SZtools also includes other nodes.
Thank you to lrzjason/T2iTraining for their work.
7. The homepage of AIGI's Bilibili for Teacher Mangosteen: https://space.bilibili.com/158424637?spm_id_from=333.1007.0.0