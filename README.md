# tinynn

玩具推理框架，写着玩，以fp16精度或者混合精度运行。短期的计划能跑简单的CNN，数据集为手写数字识别。长期目标应该能够跑ResNet，数据集为imageNet，终极目标能够跑复杂的网络。

参考：

ncnn

ncnn with cuda

paddle-Lite

## 计划

* ~~本月（2021.3）应该测试完absval_cuda算子。~~
* 之后计划，完成gemm_cuda，convolution_cuda等算子，并做些优化实践。
* 2021.4月，5月 测试完innerproduct_cuda算子，并且对kernel做些优化。
## 进度

2021.3.13

Mat（host），cudaMat（device）数据结构目前初步完成，涉及host端与device端的通信。

2021.3.24

算子absval测试完毕。

2021.4.18

算子innerproduct部分测试完成。