# VehicleClassify
车辆图片分类

使用dataset包中的tf_convert将图片数据转化为tfrecord格式，注意修改路径，图片路径参考：
imgdir/labeldir1/1.jpg..
      /labeldir2/1.jpg..
      
进入net_alexnet包中，修改alexnet_train中相关路径，然后执行即可进行训练
