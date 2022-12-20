## How to run the code (setup):
1. Downgrade tensorflow to 1.15.3 (make sure to specify .3 of 1.15.3, this is the only known version that works)
2. `git clone https://github.com/tensorflow/models.git`
3. Put infer_and_vis.ipynb under models/research
4. Replace the eval.py in models/research/deeplab with the eval.py in this folder
5. In models/research run ```bash export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ``` everytime you start a new terminal session or add it into ~/.bashrc

## To train a quantized model using pretrained weights: 
1. Download the pretrained weight by 
wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
tar -xf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz

2. In models/research/deeplab/datasets run:
bash ./download_and_convert_voc2012.sh

3. Create the directory for training log

4. In models/research/ run:
```bash
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=3000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --train_crop_size="513,513" \
    --train_batch_size=8 \
    --base_learning_rate=3e-5 \
    --dataset="pascal_voc_seg" \
    --quantize_delay_step=0 \
    --tf_initial_checkpoint=${PATH_TO_TRAINED_FLOAT_MODEL} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```
You can adjust training steps by modifying the training_number_of_steps flag. A very small learning rate such as 3e-5 by default is recommended.

## To visualize the inference result:
1. If your checkpoint directory does not contain checkpoint (not .ckpt, the checkpoint log for the directory named "checkpoint") create one and 
follow this link to document your checkpoint https://github.com/tensorflow/models/issues/4671#issuecomment-402008892. Note that the path specified must be absolute path

2. Create directories for your evaluation and visualization

3. Run the code in infer_and_vis.ipynb. The scripts it calls can only be directly called and cannot be copied and pasted in notebook and run due to tensorflow flag issues. Sometimes, you might have to manually interrupt the cell that calls vis.py after it finishes visualizing each image


## Results
We quantized the model on PASCAL VOC 2012 train_aug dataset. While we cannot obtain the mIOU with eval.py as for some reason the result we got is always a NaN, according to [tensorflow research team](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/quantize.md), they were able to obtain mIOU 0.7426 with quantized 8-bit models. We also did not observe a significant improvement of throughput.

| Bit width | mIOU | Throughput |
| --------- | ---- | -----------|
| 32 | 0.7532 | 944 |
| 8 | 0.7426 | 990|

Here are some of the samples of the segmentation results. The result order for each sample, from left to right, is original image, quantized model, unquantized model.

![Alt Text](https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/deeplab-Quantization/DeepLab-Quantization/deeplab_result1.png)

![Alt Text](https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/deeplab-Quantization/DeepLab-Quantization/deeplab_result2.png)

![Alt Text](https://github.com/Antwi007/Final-Project-Semantic-Segmentation/blob/deeplab-Quantization/DeepLab-Quantization/deeplab_result3.png)

