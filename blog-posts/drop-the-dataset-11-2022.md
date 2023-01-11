---
title: 'Drop the dataset!'
metaTitle: ''
metaDesc: ''
socialImage: 'images/drop-the-dataset-11-2022/1.jpeg'
date: '2022-11-15'
tags: 
---

_How to escape the limitation imposed by a dataset when dealing with real-life machine learning applications_

# do you want your machine learning to work?

If you have worked on machine learning in the recent years, you are familiar with a dataset, a large set of samples, pairs of inputs and desired outputs.

The goal of this post is to make you think next time you use a dataset, and setting us on course to drop them all entirely!

![](../images/drop-the-dataset-11-2022/1.jpeg)

an object categorization dataset from images

# drive your dataset

Say you wanted to make some machine learning algorithm that helps your car to drive automatically, or maybe just warn you in case you are a distracted driver. Say you start with this task: detect traffic signs and pedestrian. Both important in case a driver lost focus, as the vehicle capable of detecting them can alert you. Say we use a camera to detect the targets.

In standard supervised machine learning you would need a dataset, a set of images and possibly labels. You can organize each target category in a folder of images:

-   signs: stop, yield, speed limits, …
-   people / pedestrians

Then you train your algorithm, maybe a neural network, to recognize these target categories. Needless to say this example of driving assistance with traffic signs and people detection is a true example in the history of modern machine learning and automotive applications. This is what MobilEye started doing in 1999!

History will tell us that this approach did not work well, because in an automotive camera view there is much more than traffic signs and people. There are other important categories of targets we should detect, like other cars, vehicles, cycles. And also roads, lane markings, road work, random obstacles on the road (I once had to run over a ladder in a highway…). You see where this is going. Your dataset needs to include all sort of categories. Actually you need to include ALL categories of objects, because god knows one day you will find it outside your windshield!

So you may be tempted to just create a giant dataset with all categories of things that one can find on the road. And in machine learning history this is precisely what has happened since the creation of MNIST in 1998 and ImageNet in 2006 and even much before ([https://arxiv.org/abs/1905.05055](https://arxiv.org/abs/1905.05055))!

![](../images/drop-the-dataset-11-2022/2.jpeg)

have you included this in your dataset?

Well the issue is… you cannot possibly have a dataset of ALL objects you will ever see, because some objects that you will need to see are not even created yet, or are a combination of other object that you have not seen before.

And one popular way to train and create massive dataset for autonomous driving today is to train on synthetic data, like the one you can get from a driving video game. It does make it easier now that we have super-realistic graphics and rendering techniques, but it does not save us from detecting new and weird things we have not seen before if we did not have those in our simulator!

![](../images/drop-the-dataset-11-2022/3.png)

real or fake? ([https://blogs.nvidia.com/blog/2021/04/12/nvidia-drive-sim-omniverse-early-access/](https://blogs.nvidia.com/blog/2021/04/12/nvidia-drive-sim-omniverse-early-access/))

It gets even worse — and this is why I am writing this post! If you want to make self-driving cars, not only you will have to recognize a myriad of objects, but you also need to plan and understand the behavior of a myriad of other free agents that are sharing the road with you. You have to deal with a infinite number of combination of events that could occur on the road, most of the them “normal” driving scenarios, but occasionally rare events that no dataset will ever contain. For example: a strong rainpour that impedes you from seeing the road, snow covering the border between roads and sidewalks, a car in front of you tipping and starting to roll over, a road bridge terminated by a earthquake, a flying item carried by a strong wind. You get the idea: extreme events that you do not often see yourself, but that millions of drivers together end up seeing many thousands of times, and definitely something that our self-driving algorithm should know!

> This is the reason why self-driving cars are not here yet!

And we are there now in a time of crisys for self-driving cars: many car companies that started working on driver assistance and self-driving 5 years ago, most thinking it would be a few years job, have recanted their story, after they realized that trying to train supervised system for every possible circumstance is not possible. I was also optimistic about this field and did not fully comprehend these problems 5–10 years ago.

_What is the problem? How do we solve this?_ Even us daily driver have not seen every single object in the world, yet if we see anything on the road in front of us we break, even before recognizing what that is. Why? Why can we find the road in the snow sometimes, when all becomes just a flat white surfaces? Why can we sometimes react to millisecond events we have not seen before?

We call it “common sense”, whatever that means, but really it is a form of intelligence: to be able to predict danger and situations we have not encountered before. We cannot build this with a dataset, because there is no dataset!

![](../images/drop-the-dataset-11-2022/4.png)

understand any of this?

# Tables everywhere

I started with the previous example of a self-driving car because it is glamorous, but there is a much easier example that comes close to our daily lives: understanding documents with machine learning. I mean reading papers and documents designed for human consumption, but with an algorithm.

Say you want your machine learning algorithm to extract data from a table, plot or a diagram in a paper, so that later you can ask a question, to interpret it. Seems straightforward ah?

> today we cannot do this!

We do not have machine learning tools today that can understand and read every kind of printed tables, for example! There are just too many types of tables, with too many ways of creating columns and rows of data and delimiters and color schemes and tables formats, tables in tables, etc. You can also see that it is the same for schematic diagrams and also for scientific plots. We can read them easily ourselves, but we do not have ready algorithms that can scale to recognizing and interpreting ALL possible document parts out there.

You can see a parallel theme with the previous self-driving example: we can make a dataset to train these systems with supervision, but we will never have ALL the samples, all the categories. Sometimes we do not even know what those may be! For example if I want to create a machine learning tool that turns a paper into a patent, or a folder of data into a paper, you can see how much harder or impossible it is to define a dataset for these tasks, especially at scale. The key word here is “scale”:

> a dataset will never have all the examples you will ever need

And there is the case for general robotics also, where the number of tasks and objects and tools needed to perform daily acitvity is even much greater that the one for a self-driving robot car. The problem is the same: too many things to learn, and the need to be constantly learning them as we go!

# Give up? Never!

_I believe our limitation is truly the dataset._  We are at the beginning of time in the field of machine learning, and we have enjoyed some “victories” using datasets and supervised learning so far. Do not get me wrong, some of these victories are useful and will always be. Every time we can create a dataset that is self-sufficient and can work in real-life scenarios, we should do that. Usually this happens in controlled environments where we can artificially limit the number of objects and situations that the algorithms needs to see. There are many of these situations and we should continue to exploit them. But there are also many other situations, like the examples above, where we cannot.

In addition, the dataset and supervised mind-set has been somehow stuck in a local minima for machine learning. It has made us lazy — why bother thinking about unsupervised, or self-supervised techniques, or even continual and life-long “scale” learning if we can just use a dataset and press a button?

> self-supervised and continual learning are inevitable

So if we cannot use a dataset what should we do? We will need to think of the alternatives…

# alternatives in text

For example  _self-supervision_, when possible! This is how the best natural language processing (NLP) and large language models (LLM) like GPT-x are trained today after all! They are trained to predict the next word in a sentence. One could say there is a dataset — the sentences, but the key here is that we do not need a label, we can just use the next word! Even if we do not know that word or is a made-up word we can make this work. yes we will need a “dictionary” in some cases, and yes we cannot learn all possible future languages, but at least we can place new items in a “bin” that is unlabeled, and assign a label later when we can get sparse supervision.

# … and in images

In the world of images or video we can apply the same self-supervised techniques. In a video we can predict the next frames, or the position of objects in the future. In images we can occlude parts and try to predict them or reconstruct them. And many of these techniques have been studied and are indeed in use in machine learning applied to images and videos.

Looks like  _prediction_  is a key component of self-supervision, and in fact it is one of the prominent theories of how our brain works ([https://en.wikipedia.org/wiki/Predictive_coding](https://en.wikipedia.org/wiki/Predictive_coding)).

Porting predicting self-supervised techniques to document analysis is possible, as human readable documents are composed mainly of text and images (plots, diagrams, photos), but it is also not yet clear how one can use self-supervision to learn tables or diagrams today.

![](../images/drop-the-dataset-11-2022/5.jpeg)

i want to be more useful

# keep on adding

There is another problem here that hinders machine learning today and is tied to the use of dataset and supervised learning: continual learning. We still do not have good ways to continue to evolve a neural network, say, to new examples, new categories, new tasks. There are many proposed techniques, but none is capable of supporting life-long learning to the scale needed in a self-driving car or document understanding scenarions. I feel this problem is part of the same set of issues around dataset, and that if we keep on insisting on relying on dataset, we will not focus on solving them once and for all.

We need to strive to a goal of sharing trained machine learning models rather than datasets: share a trained neural network so that you can add your own training step, and share it again! This would make training efficient and models more useful for all of us. Sharing a dataset does not help us build more knowledge, they are a static snapshot in time.

# common sense

So what is “common sense” after all? Is it just our ability to predict the future? Or to generalize from what we know to what is unknown?

I believe it is up to us to evolve self-supervised techniques in ways that one day can be self-supported also! Today we can find clever ways to learn pieces of data from other pieces of the same set, or by correlating “concepts” into a common embedding space, or by predicting the next data from the ones we have just received. All of this has to come together into a contrastive system that can grow as needed, when we obtain new data. For a robot or a car, that may be all the time!

I believe the future are neural network learning systems that can learn autonomously new categories by embedding multi-modal data into “bins” that will become useful in later tasks, or that later task will refine, re-order, re-shape. This can be done by connecting any data that refers to the same “concepts” in a large knowledge graph that is somehow independent of datasets, tasks, objectives, but that is constantly refined by them. The key may just be  _contrastive learning_, or its evolution.

# See also:

[](https://culurciello.medium.com/data-that-bundles-together-is-learned-together-5db9629ac861)

## Data that bundles together, is learned together

### early days

culurciello.medium.com

[](https://culurciello.medium.com/at-the-limits-of-learning-46122b99dfc5)

## At the limits of learning

### Deep learning success lies with the promises of neural networks being able to learn from data. This is in contrast with…

culurciello.medium.com

[https://towardsdatascience.com/a-new-kind-of-deep-neural-networks-749bcde19108](https://towardsdatascience.com/a-new-kind-of-deep-neural-networks-749bcde19108?source=user_profile---------47----------------------------)

# about the author

I have more than 20 years of experience in neural networks in both hardware and software (a rare combination). About me:  [Medium](https://medium.com/@culurciello/),  [webpage](https://culurciello.github.io/),  [Scholar](https://scholar.google.com/citations?user=SeGmqkIAAAAJ),  [LinkedIn](https://www.linkedin.com/in/eugenioculurciello/).

If you found this article useful, please consider a  [donation](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Q3FHE3BWSC72W)  to support more tutorials and blogs. Any contribution can make a difference!
