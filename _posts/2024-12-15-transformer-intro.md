---
layout: distill
title: Transformer-Based Language Models
tags: transformers LMs pre-training
date: 2024-12-15
featured: false

authors:
    - name: H S V N S Kowndinya Renduchintala
      affiliations:
        name: MDSR, Adobe
# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## What is a Transformer-based LM?

Transformers are a class of artificial neural networks that are used to model sequences e.g., natural language text. In this article, we primarily focus on GPT-style transformers. At a fundamental level, given a piece of natural language text, these transformer-based LMs generate a probability distribution over the vocabulary. Upon repeatedly sampling from these distributions, new text can be generated. Hence the name *Generative* Pretrained Transformers(GPT). 

## How are transformers pre-trained?

Language Modeling is typically framed as a problem of **unsupervised distribution estimation** from a set of training examples $$ \mathcal{T}=(x_1, x_2, \dots, x_N) $$, where each $$ x_i $$ is composed of variable length sequences of tokens $$ (\mathbf{t_{i}^{(1)}}, \mathbf{t_{i}^{(2)}}, \dots ,\mathbf{t_{i}^{(N_i)}}) $$. Note that each $$ \mathbf{t_i^{(k)}} $$ is a $$ \lvert\mathcal{V}\rvert $$-dimensional one-hot vector where $$ \mathcal{V} $$ is the vocabulary of the language being modeled.

The joint probabilities over tokens can be written as the product of conditional probabilities as follows:

$$
    P(x_i)=\prod_{k=1}^{N_i}P\left(\mathbf{t_i^{(k)}} \mid\mathbf{t_i^{(1)}}, \mathbf{t_i^{(2)}}, \dots, \mathbf{t_i^{(k-1)}}\right)
$$

Language Models (consisting of parameters $$ \mathbf{\Theta}=\{\theta_i\}_{i=1}^{P} $$ where $$P$$ can be of the order of few billions), are trained to maximize the (log-)likelihood of text in their training set $$ \mathcal{T} $$ i.e., they try to estimate $$ \mathbf{\Theta^*} $$ where

$$
    \mathbf{\Theta^*}= \underset{\mathbf{\Theta}}{\mathrm{argmax}} \log \prod_{i=1}^{N}P(x_i;\mathbf{\Theta}) 
    = \underset{\mathbf{\Theta}}{\mathrm{argmax}}  \sum_{i=1}^{N}\log P(x_i;\mathbf{\Theta}) 
$$

$$
    =\underset{\mathbf{\Theta}}{\mathrm{argmax}} 
    \sum_{i=1}^{N} \sum_{k=1}^{N_i} \log P\left(\mathbf{t_i^{(k)}} \mid\mathbf{t_i^{(1)}}, \mathbf{t_i^{(2)}}, \dots, \mathbf{t_i^{(k-1)}};\mathbf{\Theta}\right)
$$

$$
    \mathbf{\Theta^*}=\underset{\mathbf{\Theta}}{\mathrm{argmax}} 
    \sum_{i=1}^{N} \sum_{k=1}^{N_i} \mathbf{t_i^{(k)}} . \log \mathbf{y_i^{(k)}}(\mathbf{t_i^{(1)}},\mathbf{t_i^{(2)}},\dots,\mathbf{t_i^{(k-1)}},\mathbf{\Theta})
$$

Here, $$ \mathbf{y_i^{(k)}}\left(\mathbf{t_i^{(1)}},\mathbf{t_i^{(2)}},\dots,\mathbf{t_i^{(k-1)}},\mathbf{\Theta}\right) \in \mathbb{R}^{\lvert\mathcal{V}\rvert} $$ is the vector of probabilities that are predicted by the model for the token $$ \mathbf{t_i^{(k)}} $$, given previous tokens in the input and the model parameters $$ \mathbf{\Theta} $$. In the above equation, $$ . $$ represents inner product and logarithm is element-wise. The negative of the above objective is also called as Cross-Entropy Loss.

**Remark** Note that if you give a transformer $$N_i$$ tokens, it predicts the next token for ***each*** of the $$N_i$$ prefixes i.e., it produces $$N_i$$ predictions. But why? Turns out it is easier to make one that does this. Moreover this also makes training more efficient, because you get $$N_i$$ bits of feedback rather than just one.

**Remark** Also note that we make the transformer have ***causal attention*** (a.k.a. autoregressive) i.e., it can move information only forwards in the sequence. The prediction of what comes after $$K$$ tokens is only a function of the first $$K$$ tokens. 

The key takeaway is that transformers are *sequence modeling engines*. They do the same processing in parallel at each sequence position, can move information between positions using attention, and conceptually can take a sequence of arbitrary length (not actually true as we see later)

**Inputs to a transformer** The inputs to a transformer are a sequence of tokens. Each token is a $$\lvert\mathcal{V}\rvert$$-dimensional one-hot encoded vector.