# SEIFNet
Abstract:
Remote sensing (RS) images change detection (CD) plays a crucial role in monitoring surface dynamic. However, current deep learning (DL)-based CD methods still suffer from pseudo changes and scale variations due to inadequate exploration of temporal differences and under-utilization of multiscale features. Based on the aforementioned considerations, a spatiotemporal enhancement and interlevel fusion network (SEIFNet) is proposed to improve the ability of feature representation for changing objects. Firstly, the multilevel feature maps are acquired from Siamese hierarchical backbone. To highlight the disparity in the same location at different times, the spatiotemporal difference enhancement modules (ST-DEM) are introduced to capture global and local information from bitemporal feature maps at each level. Coordinate attention and cascaded convolutions are adopted in subtraction and connection branches, respectively. Then, an adaptive context fusion module (ACFM) is designed to integrate interlevel features under the guidance of different semantic information, constituting a progressive decoder. Additionally, a plain refinement module and a concise summation-based prediction head are employed to enhance the boundary details and internal integrity of CD results. The experimental results validate the superiority of our lightweight network over 8 state-of-the-art (SOTA) methods on LEVIR-CD, SYSU-CD and WHU-CD datasets, both in accuracy and efficiency. Also, the effects of different types of backbones and differential enhancement modules are discussed in the ablation experiments in details.

![Fig1](https://github.com/lixinghua5540/SEIFNet/assets/75232301/3149f35a-4cca-4111-b03f-17492bf82cef)
