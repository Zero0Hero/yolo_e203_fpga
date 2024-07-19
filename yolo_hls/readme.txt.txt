defines.h 为网络定义，各个层大小
layers.cpp 为网络的各个层定义，包括top函数定义，卷积层定义，下采样，RGB565转换等等及加速指令
pic_fpga.h yolo_weights.h为测试图片用例以及 yolo权重
yolo_int8.cpp 为testbench文件
dpconv.cpp 为对比不同卷积计算量 加速时间等

通过vitis HLS打开本文件夹，源文件 defines.h  layers.cpp, top function为 yolo_net，仿真文件为 yolo_int8.cpp，芯片为xc7k325t