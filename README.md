# attention-network
PyTorch Implementation for Global and Local Knowledge-Aware Attention Network for Action Recognition

Convolutional Neural Networks (CNNs) have shown an effective way to learn spatiotemporal representation for action recognition in videos. However, most traditional action recognition algorithms do not employ attention mechanism to focus on essential parts of video frames which are relevant to action. In this paper, we propose a novel global and local knowledge-aware attention network to address this challenge for action recognition. The proposed network incorporates two types of attention mechanism called Statistic-based Attention (SA) and Learning-based Attention (LA) to attach higher importance to the crucial elements in each video frame. As global pooling models capture global information while attention models focus on the significant details, to make full use of their implicit complementary advantages, our network adopts a three-stream architecture, including two attention streams and a global pooling stream. Each attention stream employs a fusion layer to combine global and local information and produces composite features. Furthermore, global-attention regularization is proposed to guide two attention streams to better model dynamics of composite features with the reference to the global information. Fusion at the softmax layer is adopted to make better use of the implicit complementary advantages between Statistic-based Attention, Learning-based Attention, and Global Pooling streams and get the final comprehensive predictions. The proposed network is trained in an end-to-end fashion and learns efficient video-level features both spatially and temporally. Extensive experiments are conducted on three challenging benchmarks, Kinetics, HMDB51 and UCF101 and experimental results demonstrate that the proposed network outperforms most state-of-the-art methods


# Visualization
![image](https://github.com/ZhenxingZheng/attention-network/blob/master/dribble_20191226122506.gif )
![image](https://github.com/ZhenxingZheng/attention-network/tree/master/visualization/zheng8.jpg )


Comparison with SOTA 
![image](https://github.com/ZhenxingZheng/attention-network/tree/master/visualization/zheng10.jpg )
The pictures in the bottom row are copied from RSTAN [1]


# Coming soon
Details will be introduced soon.

# Reference
[1] W. Du, Y. Wang, and Y. Qiao, “Recurrent Spatial-Temporal Attention Network for Action Recognition in Videos,” IEEE Trans. Image Process., vol. 27, no. 3, pp. 1347–1360, 2018.
