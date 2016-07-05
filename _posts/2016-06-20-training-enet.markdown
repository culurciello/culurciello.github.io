---
layout: post
title:  "Training ENet on ImageNet"
date:   2016-06-20 10:54:01
categories: tech
---

Training a novel network on the [ImageNet dataset](http://image-net.org/challenges/LSVRC/2012/index) can be tricky. Here are some guidelines to make your model train faster and help you design better models.

This work is based on the model in our recent paper [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147). ENet network model is given [here](https://github.com/e-lab/ENet-training/blob/master/train/models/encoder.lua).

An easy way to train a neural network model on ImageNet is to use [Torch7](http://torch.ch/) and [this training script](https://github.com/soumith/imagenet-multiGPU.torch) from (Soumith Chintala)[https://github.com/soumith]. Another great training script, also deriving from this one, is [here](https://github.com/facebook/fb.resnet.torch).

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

We used Soumith training script invoked with the following command:

```
th main.lua -data /media/SuperSSD/ILSVRC2012 -backend cudnn -netType enet -nGPU 4 -batchSize 128 -nDonkeys 12
```

To justify these choices: we have 4 GPUs in our machine and 6 processors with 12 total threads, so we use 12 Donkeys to load data. That makes a big difference in loading data and overall time. Then you will get the following results:

![](/assets/enet/v23.png)

As you can see ENet V2 trains slowly, because the training script use a fairly conservative learning rate (LR) and weight decay (WD). 


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

See how long training is flat in the 1st and 2nd regime! Too much wasted time.

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

   features:add(initial_block) -- size of 112x112
   features:add(nn.JoinTable(2))
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))
   features:add(bottleneck(16, 64, true)) -- size of 56x56
   for i = 1,4 do
      features:add(bottleneck(64, 64))
   end
   features:add(bottleneck(64, 128, true)) -- size of 28x28
   
   -- pass 1:
   features:add(cbottleneck(128, 128))
   features:add(dbottleneck(128, 128))
   features:add(wbottleneck(128, 128))
   features:add(xdbottleneck(128, 128))
   features:add(cbottleneck(128, 128))
   features:add(xxdbottleneck(128, 128))
   features:add(wbottleneck(128, 128))
   features:add(xxxdbottleneck(128, 256))
   features:add(bottleneck(256, 256, true)) -- size of 14x14

   --pass 2:
   features:add(cbottleneck(256, 256))
   features:add(dbottleneck(256, 256))
   features:add(wbottleneck(256, 256))
   features:add(xdbottleneck(256, 256))
   features:add(cbottleneck(256, 256))
   features:add(xxdbottleneck(256, 256))
   features:add(wbottleneck(256, 256))
   features:add(xxxdbottleneck(256, 512))
   features:add(bottleneck(512, 512, true)) -- size of 7x7


   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
```

As a result (purple top left plots), you can see ENet V3 training faster, about 2x faster! Also it does a bit better in performance, given the model now has more output features!

Then we read [this paper](https://arxiv.org/abs/1606.02228) suggesting that linear learning rate updates may be better. So we tried this in ENet V6, basically identical to V3. Here are the results (green plots):

![](/assets/enet/v236.png)

The function to update the weight is given below:

```lua
local lr, wd
local function paramsForEpoch(epoch)
      if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified
         lr = opt.LR
         return { }
      elseif epoch == 1 then
         lr = 0.1
         return {}
      else
        lr = lr * (1-epoch/opt.nEpochs) --  math.pow( 0.9, epoch - 1)
      end
     local regimes = {
         -- start, end,     WD,
         {  1,     18,   5e-4, },
         { 19,     29,  5e-4  },
         { 30,     43,    0 },
         { 44,     52,   0 },
         { 53,    1e8,  0 },
     }
     for _, row in ipairs(regimes) do
         if epoch >= row[1] and epoch <= row[2] then
             return { learningRate = lr , weightDecay=row[3] }, true
         end
     end
 end
```

This is great, but as you can see not much different that the previous regimes used in V3.

Then we noticed that all ENet are ResNet-like network models, and so we looked at [this FB training script](https://github.com/facebook/fb.resnet.torch). Here they used a linear LR and a fixed WD of 1e-4. Adopting this and testing on ENet V7 gave us the red and orange plots:

![](/assets/enet/v2367.png)


This gave us the best results, and now it trains in ~ 10 epochs, which is 4x faster than what we started with. We used this learning rate function:

```lua
local lr, wd
local function paramsForEpoch(epoch)
      if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified
         lr = opt.LR
         return { }
      elseif epoch == 1 then
         lr = 0.1
         return { learningRate = lr, weightDecay=1e-4 }
      else
        lr = lr * math.pow( 0.9, epoch - 1)
        return { learningRate = lr, weightDecay=1e-4 }, true
      end
 end
```

This learning rate update was recommended by SangPil Kim.

ENet V7 is a bit different. It removed all dilated and asymmetric convolutions and instead uses ResNet-like modules. ENet V7 model is here:


```lua
   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block) -- 112x112
   features:add(nn.JoinTable(2))
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))
   features:add(bottleneck(16, 64, true)) -- 56x56

   for i = 1,5 do
      features:add(bottleneck(64, 64))
   end
   features:add(bottleneck(64, 128, true)) -- 28x28

   for i = 1,5 do
      features:add(bottleneck(128, 128))
   end
   features:add(bottleneck(128, 256, true)) -- 14x14

   for i = 1,5 do
      features:add(bottleneck(256, 256))
   end
   features:add(bottleneck(256, 512, true)) -- 7x7

   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
```   

Changing WD from 1e-4 to 0 after 7 epochs did not give better results.




# a balance of power


After a few experiments, ENet V12 gave really good results. Here is the model:

```
   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   features:add(initial_block) -- 112x112
   features:add(nn.JoinTable(2))
   features:add(nn.SpatialBatchNormalization(16, 1e-3))
   features:add(nn.PReLU(16))

   -- 1st block
   features:add(ibottleneck(16, 64, true)) -- 56x56
   features:add(bottleneck(64, 128))
   features:add(bottleneck(128, 128))
   
   -- 2nd block: dilation of 2
   features:add(bottleneck(128, 256, true)) -- 28x28
   features:add(bottleneck(256, 256))
   features:add(dbottleneck(256, 256))

   -- 3rd block: dilation 4
   features:add(bottleneck(256, 512, true)) -- 14x14
   features:add(bottleneck(512, 512))
   features:add(xdbottleneck(512, 512))

   -- 4th block, dilation 8
   features:add(bottleneck(512, 1024, true)) -- 7x7
   features:add(bottleneck(1024, 1024))
   features:add(xxdbottleneck(1024, 1024))

   -- global average pooling 1x1
   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
```



With the training regime listed below, it reached 65.1% in 17 epochs:

```
local function paramsForEpoch(epoch)
      if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified
         lr = opt.LR
         return { }
      elseif epoch == 1 then
         lr = 0.1
         return { learningRate = lr, weightDecay=1e-4 }
      else
        lr = lr * math.pow( 0.95, epoch - 1)
        if epoch > 7 then wd = 0 else wd = 1e-4 end
        return { learningRate = lr, weightDecay=wd }, true
      end
 end
```

Notice this is the same function as for V7, but with a slower 0.95 instead of 0.9 as decay exponent to the learning rate.

```
local function paramsForEpoch(epoch)
      if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified
         lr = opt.LR
         return { }
      elseif epoch == 1 then
         lr = 0.1
         return { learningRate = lr, weightDecay=1e-4 }
      elseif epoch > 15 then
         lr = lr * math.pow( 0.95, epoch - 15) 
         wd = 0 
         return { learningRate = lr, weightDecay=wd }, true
      else 
         wd = 1e-4
         return { learningRate = lr, weightDecay=wd }, true
      end
 end
```

We experimented with ENEt V12 with slowing down learning again, to see if it could give better results. And it did! ENet V12 reached 68.4% accuracy in approximately 30 epochs.

# more experiments

To experiment more on ENet, we decided to see the effect of dilation, dropout using the fast training mode in order to save time.

- no dilated convolutions resulted in 64.2% in 18 epochs. This is ENet V13:

	```
	   local initial_block = nn.ConcatTable(2)
	   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
	   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
	
	   features:add(initial_block)  -- 112x112
	   features:add(nn.JoinTable(2))
	   features:add(nn.SpatialBatchNormalization(16, 1e-3))
	   features:add(nn.PReLU(16))
	
	   -- 1st block
	   features:add(ibottleneck(16, 64, true)) -- 56x56
	   features:add(bottleneck(64, 128))
	   features:add(bottleneck(128, 128))
	
	   -- 2nd block
	   features:add(bottleneck(128, 256, true)) -- 28x28
	   features:add(bottleneck(256, 256))
	   features:add(bottleneck(256, 256))
	
	   -- 3rd block
	   features:add(bottleneck(256, 512, true)) -- 14x14
	   features:add(bottleneck(512, 512))
	   features:add(bottleneck(512, 512))
	
	   -- 4th block
	   features:add(bottleneck(512, 1024, true)) -- 7x7
	   features:add(bottleneck(1024, 1024))
	   features:add(bottleneck(1024, 1024))
	
	   -- global average pooling 1x1
	   features:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
	```

- no dropout used in bottleneck resulted in 64.7% in 18 epochs for ENet V14.

- The initial bock of ENet concatenates the input and a filtered version of the input:

	```lua
	local initial_block = nn.ConcatTable(2)
	   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
	   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
	...
	```

	Adding more features (29 instead of 13) to the convolution did not have any improvements on accuracy. 64.7% with 29 versus 64.9 with 13 features in ENEt V17-18 (not reported here because similar to V13)


	Notice this block is different from ResNet, where instead they use this kind of initial block:

	```lua
	   features:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
	   features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
	   features:add(nn.SpatialBatchNormalization(64, 1e-3))
	   features:add(nn.ReLU(64))
	```

	ENet initial block is much more efficient: 13 convolutions of 3x3 compared to 64 convolutions of 7x7, or 26x times more operations. 


Compare these results to the V12 fast training mode reaching 65.1% in 17 epochs.


# Comparison to ResNet

The best ENet gave a 68.4% (V12) accuracy on test set. ResNet 18 is 69.5% and ResNet34 is 73.2 as reported [here](https://github.com/facebook/fb.resnet.torch).

ResNet 18 uses a total of 3.6 G-Ops on a 224x224 input image. ENet 1.6 G-ops on the same image. And on a 960x540 input image ResNet 18 uses 38 G-Ops, while ENet uses 16.9 G-ops on the same image. 

This means ENet is 2.24x more efficient that ResNet 18.







