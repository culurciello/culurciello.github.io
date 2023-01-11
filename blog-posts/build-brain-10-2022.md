---
title: 'How can we build an artificial brain from our knowledge of the human brain?'
metaTitle: ''
metaDesc: ''
socialImage: 'images/build-brain-10-2022/3.png'
date: '2022-09-01'
tags: 
---

![](../images/build-brain-10-2022/1.jpeg)

I have been excited about reproducing the human brain in hardware and software since I read an article on neuromorphic engineering in the mid 1990s. I read much about neuroscience and psychology and of course machine learning algorithms, always with the goal of creating an artificial brain that can sense and understand the world as we humans do, or better, hopefully way better one day!

After several years, I got frustrated when I found out that we are unable to understand the brain wiring and activity of enough neurons to really be able to reverse-engineer it. When I think about ideas and a framework to replicate the brain in software, all we have today is a growing number of ideas and experiments and conjectures and even grand visions, but nothing that is unified and cohesive, and even less on a potential strategy to tackle this problem.

On the other hand, machine learning has progressed enormously in the last 10–15 years, guided by the same goals of replicating human abilities in computing machines. 15 years ago we were dreaming of identifying a human being, their posture, their actions from a camera view. Today this is only too easy! 15 years ago we were dabbling with probabilistic chains trying to unravel written human language. Today we have conversational systems that can almost pass for fellow humans. By the way, with “machine learning algorithm” here in this post I am referring to “neural networks”.

And indeed neuroscience and psychology have been a major inspiration for machine learning, even when not enough is known about our human brain, much can be tested and tried out with computer algorithms. And so it goes, 10 years of endless trials and errors, an evolution of machine learning algorithms trying not to necessarily reverse engineer the brain, as much as trying to adapt all our ideas to solving the problem of learning with algorithms. This is a  _key point,_ while there are religious faction trying to copy exactly the human brain with certified “brain approved” branding, the goal of many other has simply been: let us use any knowledge we have to engineer an artificial brain, be that a good model of our human brain or not!

Here I want to describe a framework for intelligence and learning that I call “**Mix-Match**”. Mix-Match is an algorithm and knowledge acquisition framework that is scalable and can explain how one could design the core knowledge graph acquisition in an artificial brain.

First I will describe what artificial and real brain concepts are the foundation of Mix-Match below. I will describe how artificial and real learning are similar (together) or different (apart).

![](../images/build-brain-10-2022/2.gif)

# Together

Yet I can see there are many really strong ideas from neuroscience and psychology that are one of the fundamental reason why machine learning got so good recently. I will discuss some with you below.

## Neural networks

Well hello! Neural networks are at the core of machine learning today, powering the most sophisticated learning machines we have! Artificial neural networks or ANN, as they called them, are real-valued output neurons that are modeled after biological neurons. They are the fundamental building block of computation and memory in the brain, both real and artificial. They have the ability to take multiple inputs weighted by learned kernels, and can the combine them into a non-linear output. Non-linearity is important to create any complex functions, and possibly unavoidable in real brains.

## Feed-forward visual system

Neural networks are what they are today because of many people, but if I had to pick a moment it would be because of the work of Yann LeCun in mid 1990s, where they devised a model of the human visual processing system in the form of LeNet5, the very first convolutional neural network. This network was used to learn to identify handwritten characters, and was a breakthrough not only because of its architecture, but also because it advanced gradient descent learning techniques, the core of most machine learning today. LeNet5 was modeled after the mammalian visual system processing information via simple and them more and more complex cells, extracting a hierarchy of information. LeNet5, gradient-descent, and their ability to learn and perform inspired the artificial neural network revolution of the 2010s — today.

## Cortex and micro-columns

One of the most delightful ideas is that the human neocortex is composed of the same basic circuit repeated ad libitum: the cortical micro-circuit. It does make sense to some extent: the neocortex is a flat disk of 2–4 mm of tissue, and it structure seems to be composed on many “micro columns”.

In terms of neural circuit connectivity, these micro-columns are a few layers deep, say 6, but definitely not very deep like some modern neural networks. One can thus think of a “shallow” network that is well connected to a large number of similar units.

What really inspired me about cortical micro-circuits is not just that we could break down understanding of the neocortex into a smaller step: understanding the columns. But also it reminded me of many processes in parallel requiring the concept of “attention” to sort them into useful bits.

## Attention

Attention is a key concept in psychology and neuroscience because it is the process that allow us to focus on some information that is important rather than other information that is not.

Attention in neural networks arrived later than we expected, say in around year ~2016. In reality any neural network performs attention: because the weights are in fact selecting specific signals from an array of inputs.

But really the key innovation in artificial neural networks arrived with the Transformer model in 2017. This model was using attention for almost everything, and it inspired the use of neural attention for many tasks, starting from natural language processing to vision and more.

If you look at the neural attention circuit with attention (pun intended!) you will notice it may very well be a good approximate model of cortical column circuitry. Its multi-headed attention module, in particular, is a good suitable model of how a column would process information, and find connection between sets of data. Of course it is not a 1–1 model comparison, it will never be! But to me it resonates: artificial neural attention == cortical columns.

![](../images/build-brain-10-2022/3.png)

# Apart

And there are many areas in which biological and artificial learning differ. After all why should they be the same, they are based on completely different foundational computing substrates (technologies?).

## Learning

Learning algorithms in artificial and real neural networks are clearly different. They are different in learning algorithms, continual capabilities, scope and architecture.

Artificial neural network use back-propagation and gradient descent algorithms to learn from examples. Back-propagation creates an error data-point at the output of a neural network, and that error is then used an propagated back throughput the network to adapt neural weights. Real neural network learning algorithms are not well understood, and often it is referred to Hebbian learning as a potential candidate. Hebbian learning works like this: it strongly connects neurons that fire in short succession, and depresses connections that happen non-causally or long after a cell spikes. It is normal that learning algorithms can differ, after all real and artificial neurons live in different media, with different characteristics and opportunities. If artificial network can use metal and well-insulated circuits to propagate signal farther away than real ones, then why not use that ability?

Continual learning is the ability to continue to learn at every instant and retain previous important information. So far we are in the dark on how to achieve this in artificial neural networks, and it has to do with the current supervised learning algorithms we use. I suspect that large-scale contrastive techniques, for example, have the ability to learn continuously — but in a world where they are stimulated by constant inputs and examples of all kinds. A bit like us on a daily basis. We do not yet know how to achieve the continual and life-long learning that powers our real brains, but we have an opportunity to keep searching and exploring a way to do so. This opportunity is Mix-Match, as described below. This new knowledge gathering architecture can in fact continue to learn indefinitely.

Scope and architecture of artificial neural network cannot yet take advantage or billions of years of evolution, and so machine learning researchers are left with the tedious search by trials and errors. This is a depressing proposal, but let us face the facts: we really do not yet know how to connect these artificial blocks to create anything intelligent of capable or solving multiple tasks and learning continuously, but Mix-Match can help, as we will see below. But one positive comes out of this: it allows us to refine and define the blocks that are most promising and create neural architectures that can solve low-hanging fruits. And we have done this for the last 10 years, slowly creating neural networks that conquered image categorization, natural speech and textual language, photo-realistic image creation, playing Go and other games, driving your car. Sure they are all disparate neural networks that only work on one task each, and we do need to work further on multi-modal, multi-tasks networks. The real issues there is that creatures are embodied in a set of sensors and physical configuration, and our neural networks are still living without a body, most incapable of movement, still residing inside our computer processors. This is a huge limitation to learning opportunities also due to the fact that we are far from being able to devise an artificial brain for artificial embodies entities (robots). Maybe the closest example we have today is an autonomous car — an opportunity!

## Inhibition

Real neurons use inhibition to depress the activity of neighbors and inhibitory neurons outputs are in fact a vast majority of a neurons outputs. This is due to the fact that spiking neurons cannot output negative values, as negative spike rates do not make sense. Instead, artificial neural networks  _can_  have negative outputs due to the range of the output numerical format and precision.

## **Spiking**

Real neurons spike because that is the best way to send signal across many centimeters of leaky goo. Evolution discovered that cells that can spike are able to communicate to many more neurons and thus allow attention networks to be more efficient and process a much larger amount of data. Instead artificial neurons live in silicon and metal, and their conductors are well insulated, to the point that even sending analog voltages over many centimeters is possible. And digital neurons are even better: they can transmit data without errors across the world and to far space! So there we go — we will use this ability and do away with pulsating neural networks!

![](../images/build-brain-10-2022/4.jpeg)

# Joining Forces: Mix-Match

In my opinion, the the concept that connects both artificial and real brains and has the potential to be a key ingredient to building brains and intelligence, is the concept of attention and learning to co-locate knowledge nuggets or events — I call this architecture  **Mix-Match**.

I mentioned above that multi-headed attention is the circuit we can use to address multiple applications. In fact today, artificial neural attention a la Transformer is the closest we have to an universal neural network architecture that can make sense of heterogeneous types of data. It is in fact an architecture that addresses: text, speech and vision together in one unified neural processing solution. Artificial neural attention is maybe the neural architecture diagram that comes closer to cortical columns, as it represents a  _lego-block_  of artificial neural architecture, one that replicates in the same way that our cortex is composed of many many cortical micro-circuits. Artificial neural attention is the building block of a large artificial brain that uses the same principle to sense and process all kinds of information, abstracting away the sensing domain and the format of the inputs data, and rather focusing on understanding it and correlating it that what is already known. In fact attention is a way to identifying inputs by locating them in our knowledge space, a space designed and constructed by co-locating data!

**Mix-Match**  is in my opinion the best theory of the brain and intelligence we have today. This framework is still in development and is still highly speculative, but it is important to mention it early on so we can later claimed we predicted the future! Co-locating information is simply connecting information that occurs close in space or time in the same space. It is like giving it a “name” in language. Imagine an “event” where you see a person stomping their foot and making a peculiar noise. And suppose someone observes this and says “they are stomping!” Now in the Mix-Match co-locating space, all three events: (1) the noise, (2) the video feed and (3) the words uttered because of the event will all combine into a tight embedding space. now we have co-located three sensing modalities into one. Suppose you do this for all experiences in your life, all data you ever come to sense or perceive.

> This is Mix-Match!

It is a simple learning architecture where knowledge is aggregates in its raw multi-modality, and where all components of any knowledge events come together. Modalities  _Mix_  and then are  _Matched_  to form a specific point in the embedding space.

Mix-Match is a core ingredient for an artificial brain. It is the foundation on top of which multiple abilities can arise. It supports the development of complex learning connecting the myriad of events that we witness in our lives, and can illustrate how we interlink experiences across sensory modalities, space and time.

A summary of Mix-Match properties are:

-   A common architecture for all data and knowledge — based on a large scale attention architecture
-   Continual learning is part of this system — since it expands the embedding space to include all new events without erasing previous ones
-   Application can be supervised — learning of abilities on top of the knowledge graph is a matter of connecting examples to points in the embedded space

# Final words

I hope this inspires you to build an artificial brain and learn all you can from our existing brain. There is no need to refute artificial neural networks today because they are not what we want, or as capable as our brain. Also there is no need to worry if we are unable to study our own brain today, if we do not have the tools or the ability to spy inside many many neurons as we go about our day. One day we will — maybe with Mix-Match. Maybe you can help us on that front!

# References:

Just a few references here, as they are too many.

[Neuromorphic vision chips, Analog circuits based on resistive networks emulate the behavior of the vertebrate eye, detecting edges…](https://ieeexplore.ieee.org/document/490055)

[On Intelligence (Book) by Jeff Hawkins](https://numenta.com/resources/on-intelligence/)

[From Neuron to Brain](https://www.goodreads.com/en/book/show/940331.From_Neuron_to_Brain)


[Vision Science by Stephen E. Palmer](https://www.penguinrandomhouse.com/books/655682/vision-science-by-stephen-e-palmer/)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Co$^2$L: Contrastive Continual Learning](https://arxiv.org/abs/2106.14413)


# About the author

I have almost 20 years of experience in neural networks in both hardware and software (a rare combination). See about me here:  [Medium](https://medium.com/@culurciello/),  [webpage](https://culurciello.github.io/),  [Scholar](https://scholar.google.com/citations?user=SeGmqkIAAAAJ),  [LinkedIn](https://www.linkedin.com/in/eugenioculurciello/), and more…

If you found this article useful, please consider a  [donation](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Q3FHE3BWSC72W)  to support more tutorials and blogs. Any contribution can make a difference!
