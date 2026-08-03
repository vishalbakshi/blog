---
title: Translation Labor
date: "2026-08-02"
author: Vishal Bakshi
description: It takes translation labor to collaborate well on a cross-functional team. It's the worthwhile cost of learning and building from other perspectives! In this post I share which technical skills I think are critical to perform this labor well.
filters:
   - lightbox
lightbox: auto
categories:
    - Career
---

It takes *translation labor* to collaborate well on a cross-functional team. It's the worthwhile cost of learning and building from other perspectives\!

## What is translation labor?

It’s a term I made up for this phenomenon. I like the sound of it. Comment below if there’s a better term\!

translation labor (noun)

*the work of mapping between two or more people's understanding of a problem or solution.*

Related technical skills: 

\- Empathy: viewing a situation/problem/task/solution from someone else's perspective.

\- Thoughtfulness: accounting for that perspective when you collaborate with them.

\- Asking questions: sensing that a key detail is missing from an SME's explanation

\- Voicing misunderstandings: sensing that two or more people are contradicting each other.

\- Holding space for others: letting others finish their (strange or off-putting) thought. Engaging with it in good faith.

## Illustrative Examples

### Empathy and Thoughtfulnes

Goal: we’re curating an eval set from the prod database so our SMEs can perform [error analysis](https://hamel.dev/blog/posts/evals-faq/why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed.html) 

Personnel Involved:

* You:  
  * own the evals training for SMEs  
  * have prod db access  
  * have looked at prod data in detail  
  * are aware of all project milestones and timelines.  
* Modeler B:  
  * experienced but new on this workstream  
  * focused on model fine-tuning and agent experiments for the release  
  * no prod db access  
  * not involved in evals training.  
* Tech Lead C: most experienced with user issues in prod.

Timeline: We need the dataset ready in a few days so you can train/assist the SMEs with evals using real data. The next version of the app will release in 4 weeks.

Your task: Ask Modeler B to document their ideal eval data set mix and iterate until we get approval from tech lead C.

Let's pause here for a second. 

*What would you include in your message?*

I would want to make sure that we communicate:

* Since B’s new on the project  
  * previous evals data set mix description  
* Since B doesn’t have db access  
  * database credentials  
  * sample working queries  
  * a note on the limitations of what’s included in the db tables  
* Since you’re managing the evals training  
  * a reminder on the timeline  
  * a reminder not to presume failure modes when sampling traces  
* Since the tech lead’s the most experienced with user issues in prod  
  * pinging C

These are small, meaningful ways to expect someone's needs to help them help you. And they compound fast.

### Voicing misunderstandings by asking questions

You and three other engineers are onboarding onto a project. The two most experienced engineers are leading the session. One of them uses “features”, ‘signals”,  “layers”, and “modules” as synonyms. The other one doesn’t. You assume that they both agree and that those four terms map to the same code. You have a nagging feeling they don’t. Meanwhile the complexity of the project is overwhelming you as further assumptions compound. They finish walking through a class definition for one of the signal’s layers. You jump in.

*So, to confirm: all modules layers. all layers are only for signals and features, and features and signals are the same thing?*

*Oh no, not at all. Here, take a look at the architecture diagram again, it lays it all out, here’s the link.*

*Thanks for asking that. I was getting confused as well.*

### Holding space for others

You’re a senior engineer. We’re at standup. An engineer says they have found a bug in your data pipeline. The new product’s data isn’t showing up in the batches during the forward pass. Two dataframes from different sources are inner-merged. One of them has new product data and the other doesn't, so those rows get dropped before it reaches the model. They’ve spent a week debugging this and are itching to get to the modeling work. You’ve been debugging this with them and are pretty confident about this one.

*Why don't we add the new data in both sources upstream of this instead of one?*

The PM jumps in.

*No, we don't want this product’s data with the other products data--*

You want interrupt with the following

*Y'all, we gotta stop creating separate sources for the same thing. It's going to bloat an already bloated codebase. I've been cleaning this up all week.*

but you don’t, and instead say

*Why?*

*…because this product isn't modeled like the other ones*.

You pause. You’re taken aback.

*Oh shoot, I didn't realize that was the case. Sorry, I've been giving the wrong advice on this bug the whole time. Okay, I'll schedule a call to fix this.*

## Is minimizing translation labor always good?**

We must always balance speed, accuracy and precision. Translation labor is no different. 

In what situations is translation labor desirable? When is it not? Ultimately it's situation dependent, but being aware of it helps me navigate my career.

