import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def reshape_transform(tensor, height=28, width=13):
    if tensor.shape[1]==365:
        # 去掉类别标记
        result = tensor[:, 1:, :].reshape(tensor.size(0),
        height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    else:
        print("SB")
        return None


def CAMPLT(model,use_cuda,path,i):
    target_layer=[model.base_resnet.base.layer4[-1]]
    # 创建 GradCAM 对象
    cam = EigenCAM(model=model,
    target_layers=target_layer)
    
    # 读取输入图像
    image_path = path
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (192, 384))

    # 预处理图像
    input_tensor = preprocess_image(rgb_img,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

    # 将图像转换为批量形式
    input_tensor = input_tensor
    if use_cuda:
       input_tensor = input_tensor.cuda()
    
    # 计算 grad-cam
    target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam = cam(input_tensor=input_tensor,
    targets=target_category)

    # 将 grad-cam 的输出叠加到原始图像上
    #visualization = show_cam_on_image(rgb_img, grayscale_cam)
    visualization = show_cam_on_image(rgb_img.astype(dtype=np.float32)/255.,grayscale_cam[0])
    #a = np.transpose(visualization, (2, 0, 1)).reshape((1,3,288,144))
    # 保存可视化结果
    #cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB, visualization)
    cv2.imwrite('/home/gml/HXC/images/ooo/'+str(i)+'cnn.jpg', visualization)
    print("")


