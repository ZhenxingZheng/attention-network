# attention-network
PyTorch Implementation for Global and Local Knowledge-Aware Attention Network for Action Recognition

Convolutional neural networks (CNNs) have shown an effective way to learn spatiotemporal representation for action recognition in videos. However, most traditional action recognition algorithms do not employ the attention mechanism to focus on essential parts of video frames that are relevant to
the action. In this article, we propose a novel global and local knowledge-aware attention network to address this challenge for action recognition. The proposed network incorporates two types of attention mechanism called statistic-based attention (SA) and learning-based attention (LA) to attach higher importance to the crucial elements in each video frame. As global pooling (GP) models capture global information, while attention models focus on the significant details to make full use of their implicit complementary advantages, our network adopts a three-stream architecture, including two attention streams and a GP stream. Each attention stream employs a fusion layer to combine global and local information and produces composite features. Furthermore, global-attention (GA) regularization is proposed to guide two attention streams to better model dynamics of composite features with the reference to the global information. Fusion at the softmax layer is adopted to make better use of the implicit complementary advantages between SA, LA,
and GP streams and get the final comprehensive predictions. The proposed network is trained in an end-to-end fashion and learns efficient video-level features both spatially and temporally. Extensive experiments are conducted on three challenging benchmarks, Kinetics, HMDB51, and UCF101, and experimental results demonstrate that the proposed network outperforms most state-of-the-art methods.



## Requirements

PyTorch 0.4.1

opencv-python

Directory Tree

```
Datasets/
  HMDB/
    dirs of class name/
      dirs of video names/
        0.jpg
  list/
    hmdb_final_split1_train.txt
      HMDB/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0 408 0
dirs of code name/
  main.py
```



## Datasets

The datasets and splits can be downloaded from

[Kinetics-600](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)

[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)




## Visualization
![dribble_20191226122506](https://github.com/ZhenxingZheng/attention-network/blob/master/dribble_20191226122506.gif)




## Models
The pretrained models on Kinetics can be downloaded from [Baidu Yun](https://pan.baidu.com/s/1mUknQibh6Xs5MryI5qwqYQ), code:oa98.




## Citation

```
@ARTICLE{9050644,  
author={Z. {Zheng} and G. {An} and D. {Wu} and Q. {Ruan}}, 
journal={IEEE Transactions on Neural Networks and Learning Systems},   
title={Global and Local Knowledge-Aware Attention Network for Action Recognition},   
year={2020},  
volume={}, 
number={}, 
pages={1-14},
doi={10.1109/TNNLS.2020.2978613},
}
```


