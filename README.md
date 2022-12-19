![]( https://visitor-badge.glitch.me/badge?page_id=antwi007.Final-Project-Semantic-Segmentation)

<div style="display:flex;">
  <img src="https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/FCN-Quantization/img1real.png" width="400" height="400">
<img src="https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/FCN-Quantization/image1segmentation.png" width="400" height="400">
</div>


# Quantizing Deep Learning Models For Semantic Image Segmentation
Welcome to our project on quantizing deep learning models for semantic image segmentation!

Semantic image segmentation is the task of assigning a label to each pixel in an image, indicating the class of object or scene that pixel belongs to. Deep learning models, such as convolutional neural networks (CNNs), have achieved state-of-the-art performance on this task, but often require a large number of parameters and computational resources to achieve this performance.

One way to reduce the computational complexity of deep learning models is through quantization, which involves representing model parameters and activations using fewer bits. This can significantly reduce the size and inference time of the model, making it more suitable for deployment on resource-constrained devices.

Our project focuses on quantizing deep learning models for semantic image segmentation, with the goal of achieving similar or better performance with fewer bits. We provide code and instructions for quantizing popular semantic segmentation models, such as FCN and UNet, on popular datasets like [COCO 2017](https://cocodataset.org/#download) and [The ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000502).

### Repo Structure
The root directory of this repository contains the following directories:

1. `FCN-Quantization`: This directory contains code and instructions for quantizing the FCN-ResNet50 model for semantic image segmentation.

2. `U-Net-Quantization`: This directory contains code and instructions for quantizing the U-Net model for semantic image segmentation in medical imaging.

3. `tf.yml`: Conda environment containing all libraries and versions used to run all the code in the repo.

Each directory contains a `README.md` file that provides an overview of the contents of the directory, including any dependencies and instructions for running the code.

# Results

In this section, we present the results of model quantization experiments on different datasets. For detailed instructions on how to execute the code and reproduce these results, please refer to the `README` file in the relevant subdirectory (fcn-resnet50 or u-net).

## FCN-ResNet50 on COCO Val 2017 

We quantized the FCN-ResNet50 model on the COCO 2017 Val dataset using various bit widths for the model parameters and activations. Our results showed that quantizing to 8 bits resulted in no drop in performance compared to the full-precision model, while still providing a significant reduction in model size and inference time.

| Bit width | mIoU | Model size | Throughput     |
|-----------|------|------------|----------------|
| 32        | 0.78 | 135 MB     | 3.5 images/s         |
| 8         | 0.77 | 34 MB      | 2.8 images/s         |

 We also tested the models on different hardware platforms to see how they would perform in practice. Below, we present the results of our inference time experiments on GPUs and CPUs.

 ### Inference time on GPUs
![Alt Text](https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/results_images/fcn_gpu.png)

`Tesla T4` and `Tesla V100-SXM2` were provided by google cloud platform. `NVIDIA GeForce RTX 3080 Ti` experiment was run locally. Inference speed on GPU is higher for quantized model with an average speed up of 1.25x


 ### Inference time on CPUs

![Alt Text](https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/results_images/fcn_cpu.png)


Experiments with `Intel(R) Xeon(R) CPU @ 2.00GHz` and `Intel(R) Xeon(R) CPU @ 2.30GHz` processors were run on google cloud platform. `AMD Ryzen 7 3700X 8-Core Processor` and `Quad-Core Intel Core i7` was run locally. Inference speed on CPU is higher for quantized model with an average speed up of 1.79x

### Segmentation Samples on Quantized and Non-Quantized Models 

<div style="display:flex;">
  <img src="https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/results_images/real_img.png" width="250" height="250">
  <img src="https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/results_images/quantized_img.png" width="250" height="250">
  <img src="https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/nana/results_images/real_img_normal.png" width="250" height="250">
</div>

The leftmost image is the original image, the middle is the segmenation provided by the quantized model, and the rightmost is the segmentation mask provided by the normal non-quantized model. There's no significant drop in quality of the segmentation provided by the quantized model as compared to the non-quantized model.

## Citation

Bibtex:

    @inproceedings{askarimiccai2019,
    title={U-Net Fixed Point Quantization For Medical Image Segmentation},
    author={AskariHemmat, MohammadHossein and Honari, Sina and Rouhier, Lucas  and S. Perone, Christian  and Cohen-Adad, Julien and Savaria, Yvon and David, Jean-Pierre},
    booktitle={Medical Imaging and Computer Assisted Intervention (MICCAI), Hardware Aware Learning Workshop (HAL-MICCAI) 2019},
    year={2019},
    publisher={Springer}
    }
  

