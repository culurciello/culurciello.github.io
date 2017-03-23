---
layout: post
title:  "Hardware for Deep Learning"
date:   2016-09-11 11:01:07
categories: tech
---

# I like Deep Learning ...

Deep Learning's recent success is unstoppable. From categorizing objects in images and speech recognition, to captioning images, understanding visual scenes, summarizing videos, translate language, paint, even produce images, speech, sounds and music!


# ... and I want to run it fast!

The results are amazing, and so the demand will rise. Imagine you are Google or Facebook or Twitter: after you find a way to "read" the content of images and videos to make a better model of your users, what they like, what they talk about, what they recommend, and what they share. What would you do? You would probably like to do more of it!

Maybe you run a version of ResNet / Xception / denseNet to classify user images into thousands of categories. And if you are one of the internet giants, you have lots of servers and server farms, so ideally you want to run Deep Learning algorithms on this existing infrastructure. And it works for a while... until you realize those servers that you were using to parse text, now have to do > 1 Million times the operations that have to do before to run categorization of single images. And data from use trickles down faster and faster. [300 hours of video for each minute of real life!](http://fortunelords.com/youtube-statistics/).

[Server farms consume a lot of power](https://www.cloudyn.com/blog/10-facts-didnt-know-server-farms/) and if we need to use 1M more infrastructure to process images and videos, we will need to either build a lot of power plants, or use more efficient ways to do Deep Learning in the cloud.
An power is hard to come by, so we better take the efficiency route going forward.

But data centers are only one of the areas where we need more optimized microchips and hardware for Deep Learning solutions. In an autonomous car it may be ok to place a 1000 Watt computing system (albeit that will also use battery/fuel), but in many other applications, power is a hard limit. Think drones, robots, cell-phones, tablets and other mobile devices. These all need to run on a few watts of power budget, if not below 1 W.

And there are a lot of consumer products, like smart cameras, that also need to consume little power, and may not want to use cloud computing solutions for privacy issues.

And with our homes becoming smarter and smarter one can see that many devices will need to use Deep Learning applications, collect and process data on a continuous basis.


# so... you need new hardware, ah?

So we need new hardware, one that is more efficient than Intel-Xeon-powered servers. An Intel server CPU may consume 100-150 W and may also need a large system with cooling to support the performance.

What are other options?

- Graphic processors, GPU
- field-programmable logic devices, FPGA
- custom microchips, application-specific integrated circuits, ASICs, or systems on a chip, SoC
- digital signal processors, DSP
- some other technology we may get from the future, aliens, or obscure new laws of physics


# GPUs

GPUs are processors designed to generate polygon-based computer graphics. In the recent years, given the sophistication and need for realism of recent computer games and graphic engines, GPUs have accumulated large processing powers.
NVIDIA is leading the game, producing processors with several thousand cores designed to compute with almost 100% efficiency. Turns out these processors are also well suited to perform the computation of neural networks, [matrix multiplications](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/). Notice matrix-vector multiplications are considered ["embarrassingly parallel"](https://en.wikipedia.org/wiki/Embarrassingly_parallel) because they can be parallelized with easy algorithm scaling (they lack the branching and thus do away with little [cache misses](https://en.wikipedia.org/wiki/CPU_cache)).

The [Titan X](http://www.geforce.com/hardware/10series/titan-x-pascal) is one favorite workhorse for training Deep Learning models. With more than 3500 cores, it can deliver [more than 11 Tera-flops](https://blogs.nvidia.com/blog/2016/07/21/titan-x/). More information on tested performance is [here](https://github.com/soumith/convnet-benchmarks).

The fight between Intel CPUs and NVIDIA GPUs [favored the latter](https://blogs.nvidia.com/blog/2016/08/16/correcting-some-mistakes/) because of the large amount of cores of GPUs (~3500 vs 16 of an Intel Xeon, or 32 of Xeon-Phi), offsetting the 2-3 faster speed of CPU clocks. The GPUs cores are streamlined version of the more complex (branch prediction and pipelined) CPU cores, but having so many of them enables higher level of parallelism and thus more performance.

At this time GPUs are the norm in training Deep Learning systems, be that convolutional / CNN or recurrent neural networks / RNN. They can train on large batches of images of 128 or 256 images at once in just a few milliseconds. But they consume ~250 W and require a full PC to support them, with additional 150 W of power. No less than 400 W may go into a high-performance GPU system.

This is not an option for drones, cell-phones, mobile devices, and small robots. And even in a future consumer-grade autonomous car this power budget is not acceptable.

NVIDIA is working hard on more power-efficient devices, such as the [Tegra TX1](http://www.nvidia.com/object/jetson-tx1-dev-kit.html) (12 W and ~100 G-flops/s performance on deep neural nets) and the more powerful [Drive PX](http://www.nvidia.com/object/drive-px.html) (250 W, like a Titan X).

Notice also that in the case of autonomous cars, and smart cameras, where live video is necessary, image batching is not possible, as video has to be processed in real time for timely responses.

In general GPUs deliver ~5 G-flops/s per W of power. We need to do better than this if we want mobile systems to deploy deep learning solutions!



# FPGA

Modern FPGA devices such as the ones from [Xilinx](https://www.xilinx.com/) are the Lego of electronics. One can build entire custom microprocessors and complex heterogenous systems using their circuits as building blocks. And in the recent years, FPGA started sporting more and more multiply-accumulate computing blocks. These DSP blocks, as they are called, can perform multiplications, and can be arrayed together to perform many of them in parallel.

We have been working for more than 10 years on using FPGA for neural networks. Our work started from initial pioneering work from Yann LeCun team at NYU, and in particular of [Clement Farabet](http://yann.lecun.com/exdb/publis/pdf/farabet-fpl-09.pdf). Our collaborative work produced [NeuFlow](http://snowbird.djvuzone.org/2011/abstracts/167.pdf) a complex data-flow processor for running neural networks.

During the years 2011 to early 2015, we perfected a completely new design called [nn-X](http://ieeexplore.ieee.org/document/6910056/?tp=&arnumber=6910056). This work was lead by Berin Martini and Vinayak Gokhale (from our lab). The system delivered up to 200 G-ops/s on a 4 W budget, effectively 50 G-ops/s/W, or almost 10x more than GPUs.

But nn-X suffered two main issues:
- low utilization when the fixed convolutional engines were not used
- high memory bandwidth
The first issue was due to the fact that nn-X employed fixed convolutional engines of 10x10, and when performing 3x3 convolutions, only 9% of the DSP units were effectively used. This was ameliorated later by dividing a 12x12 grid into 4x4 units of 3x3 convolvers. Unfortunately the system also needed high memory bandwidth because it did not employ data cache and required fetching inputs from memory and saving results directly into memory. As such nn-X was not able to scale and its utilization of DPS units was never above 75-80%.

Systems with similar design constrains will also be [limited](http://www.deephi.com/) in performance.

What is needed is a system with data cache that can use arbitrary groups of DPS units to effectively use close to 100% of the resources. One such system is [Microsoft Catapult](https://www.microsoft.com/en-us/research/project/project-catapult/) and our own SnowFlake accelerator (more info coming soon). Which uses Altera devices to achieve record performance in executing deep neural networks. Unfortunately this is not a commercial system, but rather one of Microsoft data-center assets, and thus is not yet available to the public. 
The Chinese Internet giant Baidu also [followed route](http://www.hotchips.org/wp-content/uploads/hc_archives/hc26/HC26-12-day2-epub/HC26.12-5-FPGAs-epub/HC26.12.545-Soft-Def-Acc-Ouyang-baidu-v3--baidu-v4.pdf).


# Custom SoC

Qualcomm, AMD, ARM, Intel, NVIDIA, are all working hard on integrating custom microchips into their existing solutions. Nervana and Movidius have or are developing integrated solutions. 
SoC can provide ~10x better performance than FPGA system on the same technology node, and more in some specific architectures.
As the power of SoC and processors becomes lower and lower, the differentiation will come from new integrated memory systems, and the efficient use of bandwidth to external memory. In this area 3D memory integrated as systems-on-a-package are a way to decrease power by at least 10x.


# DSP

DSPs have been around for a long time, and were born to perform matrix arithmetics. But to date, no DSP has really provided any useful performance or device that can compete with GPUs. Why is that? The main reason is the number of cores. DSP were mainly used for telecommunication systems, and did not need to have more than 16 or 32 cores. Their workload just did not need it. Instead GPU workloads kept increasing more and more in the recent 10-15 years, thus requiring more and more cores. At the end, since the year 2006 or so, NVIDIA GPUs were superior in performance to DSPs.

Texas Instruments continue to develop them, but we have not seen any competitive performance out of these. And many DSP have also been supplanted by FPGAs.

Qualcomm use [DSPs](https://www.xda-developers.com/qualcomm-is-optimizing-the-snapdragon-835-and-hexagon-682-dsp-for-tensorflow/) in their SoC, and they provide acceleration, but there is no enough detail to date to compare them to other solutions.


# The future

Is for us to make.

Stay interested, focused and active!
