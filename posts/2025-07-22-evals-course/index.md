---
title: Takeaways from Lesson 1 of the AI Evals Course
date: "2025-07-22"
author: Vishal Bakshi
description: In this blog post, I'm going highlight ideas that stood out to me from the first lesson and first three chapters of the course reader from the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Background

In this blog post, I'm going highlight ideas that stood out to me from the first lesson and first three chapters of the course reader from the AI evals course by Hamel Husain and Shreya Shankar. 

## Ideas that Stood Out from the First Three Chapters of the Course Reader

**Idea 1: LLMs used for intermediate tasks still need evals.** I think this is an important reminder because I often think of LLM tasks as complex or mission-critical or user-facing. However, there are smaller subtasks that happen "behind the scenes", assisting the rest of the pipeline. These tasks can be overlooked. For example you could have an LLM-assisted task of curating few-shot examples that are used for a larger prompt downstream. This task needs evals.

**Idea 2: looking at data, whether it's curating few-shot examples manually or reviewing LLM traces to classify failures, deepens our understanding of what the user wants and how the LLM fails to deliver.** What makes a good example? We have to think about objectives, content, format, tone, instruction, and output. What type of failure are we witnessing in a trace? We have to think about those dimensions, and identify specifically what the LLM failed to do and categorize it precisely. 

**Idea 3: We allow ourselves to adjust annotations or revise failure mode definitions as needed as we organize our thoughts and observations on LLM traces.** It's normal for annotation schemas to evolve after reviewing more data. I fully agree and relate to this idea. [After evaluating 1350 LLM judge scores (for LM-generated tiny stories)](https://youtu.be/FXOXoaGjntc) I improved my understanding of what makes a high quality story (in terms of creativity, grammar, context-tracking, plot, factual knowledge, and reasoning) and modified the criteria I used to evaluate those dimensions.

**Idea 4: Forcing binary decisions about whether a failure mode occurs or not produces more reproducible annotations than the Likert scale.** As someone who has advised, designed and administered a dozen or more surveys, I plead the case for binary decisions and groan at the sight of a Likert scale. What's the difference between "Good" and "Very Good"? What about "Mostly Satisfied" and "Somewhat Satisfied"? If you're asking someone to evaluate even 10s of LLM traces using a Likert scale, the cognitive load quickly fatigues the evaluator. It's much simpler to answer "Did the LLM fail to use an appropriate tone?" (Yes/No) than "How well did the LLM use an appropriate tone?" (Exceptionally Well, Very Well, Somewhat Well, Not Well at All).

Note: Allen Downey has an excellent blog post, [_The Mean of a Likert Scale_](https://www.allendowney.com/blog/2024/05/03/the-mean-of-a-likert-scale/) which tackles the challenges of summarizing Likert scale data.

## Ideas from Lesson 1

**Idea A: LLM Evaluation is the systematic measurement of LLM pipeline quality.** This is the first (and best) concise definition of evals I've come across. We want to create a system (that's reproducible and reliable) to measure the quality of an LLM pipeline. What to measure in the pipeline and how to define quality is what this course will teach us.

**Idea B: Pay attention to your prompt.** Many folks don't read the prompt they're sending to the LLM! Think about the prompt first before writing an AI-assisted prompt. Specifying your problem involves writing your prompts with care. Use an LLM where it makes sense to be used (e.g. to improve the clarity of your already specific prompt).

**Idea C: You have to wear all these hats.** To cross the gulf of comprehension (i.e. what is my data and what stories does it tell?) you must wear your "data scientist" hat. For specification, your "product" hat, and for generalization your "engineer" hat. Shreya and Hamel encourage us to move slowly as we navigate these gulfs. As Kawhi Leonard says: [slow is pro](https://x.com/patbev21/status/1884687382412132558).

**Idea D: Collecting representative samples and analyzing failure modes is the most important step in AI evals.** This is where you learn the most, is what most people skip, what has the least guidance out there in the industry. Shreya spends 75-80% of her time on error analysis. Asking the question "what would make a user unhappy?" uncovers your failure modes. 

**Idea E: The process of writing a good prompt sets the foundation for your evaluation success.** If your prompt fails to specify, in no uncertain terms, what the LLM should and shouldn't do, how can you expect to evaluate the LLM's output? "The LLM didn't structure its response as JSON"---did you ask it to? "The LLM didn't provide measurements for the recipe"---did you ask it to?

**Idea F: The process of removing errors from the LLM pipeline is a process of improving your communication to the LLM,** and this requires empathy! You have to put yourself in the user's shoes and ask: what do I want from this interaction? What will make me happy? What will make me frustrated? And then you have to relay those user needs to the LLM via the prompt. And finally you have to evaluate whether the desired experience was delivered by the LLM. This dance between playing data scientist/product manager/user/engineer/communicator makes applied AI such a fascinating field.

## Closing Thoughts

I'm swimming in tasks right now (including a new puppy!) so it's important for me to move slowly and deliberately. Taking this course at this juncture in my life is itself a process of evaluation: what do I focus my time on? How deep do I go into which assignments? What questions should I ask during office hours? The excellent structure and resources that Hamel and Shreya provided us will help ease my cognitive load, allowing me to focus on high value tasks. I don't know if I'll be able to write a blog post or make a video for each lesson, but I'll try to publish at least once a week (on average). Our HW is to write a system prompt for a recipe chatbot, so expect some of my musings on that soon!