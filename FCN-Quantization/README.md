### Follow this guide to run the code in this repository

This directory contains code to quantize and execute performance evaluation of the quantized and the normal model. Follow these steps to reproduce results or try new methods. 

### Model Download and Training 

Download the pytorch FCN-ResNet50 model:

To download the pytorch FCN-ResNet50 model, you can use the following link: [FCN_RESNET50](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet50.html). Save the model to a convenient location on your local machine, such as a folder called "models".


### Model Quantization and Benchmarking
Execute the quantize_model.ipynb notebook:

To quantize the model and check its benchmark performance against the non-quantized version, open the quantize_model.ipynb notebook in your preferred Python environment and execute the code. The notebook contains well-documented code that will guide you through the process of quantizing the model and comparing its performance to the non-quantized version.

`fcn_rn50.yaml` contains parameters for executing quantization by the Intel Neural Compressor Library. Sample code:

```from neural_compressor.experimental import Quantization, common
from neural_compressor import options
options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

quantize = Quantization("fcn_rn50.yaml")
quantize.model = common.Model(model)
quantize.calib_dataloader = common.DataLoader(ds)
quantize.eval_func = eval
q_model = quantize()
q_model.save("models/quantized/fcn-resnet50-int8.onnx")

```

### Inference and Segmentation Output
Run the model_inference.ipynb notebook:

To test the quantized model on the COCO Val 2017 dataset, open the model_inference.ipynb notebook and execute the code. The notebook contains well-documented code that will guide you through the process of using the quantized model for inference on the COCO Val 2017 dataset.


Note: Be sure to follow the instructions in the notebooks carefully and pay attention to any dependencies or prerequisites that may be required for running the code.
