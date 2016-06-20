---
layout: post
title:  "Training ENet on ImageNet"
date:   2016-06-20 10:54:01
categories: tech
---

Training a novel network on the ImageNet dataset can be tricky. Here are some guidelines to make your model train faster and help you design better models.

This work is based on the model in our recent paper [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147). ENet network model is given [here](https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua).

An easy way to train a neural network model on ImageNet is to use [this training script](https://github.com/soumith/imagenet-multiGPU.torch) from (Soumith Chintala)[https://github.com/soumith]. Another great training script, also deriving from this one, is [here](https://github.com/facebook/fb.resnet.torch).

Regardless, ENet needs to be modified to run on ImageNet images of 224x224 size. This is done by modifying a downsampling bottleneck function at the end of the last 2 modules (in the for loop). This is ENet V2:

```lua
local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block) -- size of 112x112
   features:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))
   features:add(bottleneck(16, 64, true)) -- size of 56x56
   for i = 1,4 do
      features:add(bottleneck(64, 64))
   end
   features:add(bottleneck(64, 128, true)) -- size of 28x28
   for i = 1,2 do
      features:add(cbottleneck(128, 128))
      features:add(dbottleneck(128, 128))
      features:add(wbottleneck(128, 128))
      features:add(xdbottleneck(128, 128))
      features:add(cbottleneck(128, 128))
      features:add(xxdbottleneck(128, 128))
      features:add(wbottleneck(128, 128))
      features:add(xxxdbottleneck(128, 128))
      features:add(bottleneck(128, 128, true)) -- size of 14x14, then 7x7
   end
   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
```

If you use Soumith training script, you will get the following results:

![](/assets/enet/v23.png)

As you can see ENet V2 trains slowly, because the training script use a fairly conservative learning rate (LR) and weight decay (WD). See how long training is flat in the 1st and 2nd regime! Too much wasted time.

```lua
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }
```

So for ENet V3, we decided to modify the training regime to go faster:

```lua
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     10,   1e-2,   5e-4, },
        { 11,     15,   5e-3,   5e-4  },
        { 16,     20,   1e-3,   0 },
        { 21,     30,   5e-4,   0 },
        { 31,    1e8,   1e-4,   0 },
    }
```

Also ENet V3 was modified to have more output features: 512 like ResNet 18 and 34. Here is ENet V3 model:


```lua
   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block)                                         -- 128x256
   features:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))
   features:add(bottleneck(16, 64, true))                              -- 64x128
   for i = 1,4 do
      features:add(bottleneck(64, 64))
   end
   features:add(bottleneck(64, 128, true))                             -- 32x64
   
   -- pass 1:
   features:add(cbottleneck(128, 128))
   features:add(dbottleneck(128, 128))
   features:add(wbottleneck(128, 128))
   features:add(xdbottleneck(128, 128))
   features:add(cbottleneck(128, 128))
   features:add(xxdbottleneck(128, 128))
   features:add(wbottleneck(128, 128))
   features:add(xxxdbottleneck(128, 256))
   features:add(bottleneck(256, 256, true))

   --pass 2:
   features:add(cbottleneck(256, 256))
   features:add(dbottleneck(256, 256))
   features:add(wbottleneck(256, 256))
   features:add(xdbottleneck(256, 256))
   features:add(cbottleneck(256, 256))
   features:add(xxdbottleneck(256, 256))
   features:add(wbottleneck(256, 256))
   features:add(xxxdbottleneck(256, 512))
   features:add(bottleneck(512, 512, true))


   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
```

As a result, you can see ENet V3 training faster, about 2x faster! Also it does a bit better in performance, given the model now has more output features!



