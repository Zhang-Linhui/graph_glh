result中有结果图表，
更美观的可视化见例子图片
方法：https://www.zhihu.com/question/59448796/answer/3442992217
在装pytorch的环境下，安装tensorboad后，在本目录cmd下，输入命令：tensorboard --logdir runs

代码：
main函数中通过调整model_name = 'APPNP'，这个变量来切换模型
data_process 将csv处理成npy 

模型文件夹下的模型可以问gpt

就结果而言这个数据有点问题，因为结果都差不多80%正确率