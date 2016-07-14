---
layout: post
title:  "Video summarization: why and how?"
date:   2016-07-09 08:24:01
categories: tech
---

It has been a long time, but the dream is always the same: can I get my computer to watch a video and summarize it for me?

The answer right now is: no.

# Why?

Video or story analysis requires to extract features at key-point "frames". These features encode the meaning of the scene. Multiple scenes make up a temporal sequence of events, or action performed in the portion of video or story.

Video analysis can use Deep Neural Networks (DNN or CNN) to describe each frame. If such DNN is trained with a large number of objects and categories, it may be able to create a good "representation" of the content of single frames. 

But in a video the temporal aspect is important, and for understanding actions, we will need to integrate multiple frames. Using a DNN with multiple frames as input gives little improvements over a single frame as can be seen in [this paper](http://cs.stanford.edu/people/karpathy/deepvideo/deepvideo_cvpr2014.pdf).

In order to use the temporal information of videos (with audio also, if present) we can feed the representation of individual "frames" to a Recurrent Neural Network (RNN). [Here is an example](http://arxiv.org/abs/1503.08909). RNN are designed to remember and recognize sequences, so they are perfect to understand the sequential content of video information.

Please note that "frames" are artificially created in a video by sampling at fixed intervals of time. But individual frames are highly correlated, so the information needs to be aggregated. Similarly for audio, chunks of a few millisecond of signal are aggregated into spectral ensembles.

Also not that the "representation" of a video may contain a mixture of audio and video information.

Note: the paper listed above are from George and his YouTube team (Google), which is one of the most prominent and active teams in the area of video understanding and summarization. In these papers notice the large amount of data. For using example Sport-1M, neural network were trained with more than 500M frames. That is a very large number. We need to do better than this and train with less labeled frames, and with less data in overall.



# How?


We need to show that by reading a video sequence, we can extract some meaningful “representation”.

A lot has to do with the task and dataset at hand: in applied data science, you get what you put in. 

To summarize a story, or video, one needs to create the following representation:

- actors: John
- actors characteristics: indian, black hair, tall, ...
- actions: goes to kitchen, takes a sandwich
- locations: kitchen, house…

In other words the representation is always: actor, place, action - since we live in a causal world!!!!
So this is what gets in our brain, and what should get into our neural net.

# the future

Is for us to make.

Stay interested, focused and active!
