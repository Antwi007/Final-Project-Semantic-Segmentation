### Follow this guide to Run the Code in this Repo

This directory contains code for U-Net quantization and test on [The ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000502)


### Downloading Dataset:

Download pre-processed data from [this link](https://drive.google.com/file/d/1kjc3HLVuGdMa9wBF1SHaNicH9Y-maDzZ/view?usp=sharing). You can also you use pre-processing code in the repo originally found 


### Configuring the Model using config.yaml:

 The YAML file specifies which specific configuration to use for a specific experiment. For instance, for EM dataset, to run an experiment with Fixed bi quantization with 8-bit for activation and 8-bit for Weights, you need to configure the yaml file as follows:

```yaml

UNET:
    dataset: 'neuronal_dataset' #Could be anything
    lr: 0.001
    num_epochs: 200
    model_type: "unet"
    init_type: glorot
    quantization: "FIXED" # "Normal 
    activation_f_width: 8
    activation_i_width: 8
    weight_f_width: 8
    weight_i_width: 8
    gpu_core_num: 1
    trained_model: "/path/to/trained/models/experiment_1_8_8.pkl"
    experiment_name: "experiment_1"
    log_output_dir: "/path/to/output/folder"
    operation_mode: "normal"
```

### Running code:

Run the following code in the terminal:

`python em_unet.py -f config.yaml -t UNET`

### Monitoring Training Result 

`tensorboard --logdir ./path/to/logs/em_001_3_fixed/losses/`

### Checking Model Size after Quantization

Run the notebook model_size.ipynb to find the model size after quantization. Basically calculates the number of parameters and buffer size, and outputs the total size in MB. Follow the guidlines in the repo closely. 

## Citation

AskariHemmat, MohammadHossein, et al. "U-Net Fixed Point Quantization For Medical Image Segmentation." Medical Imaging and Computer Assisted Intervention (MICCAI), Hardware Aware Learning Workshop (HAL-MICCAI) 2019. Springer, 2019.
