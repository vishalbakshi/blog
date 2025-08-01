---
title: Standout Ideas from Lesson 3 and Chapter 4 of the Course Reader from the AI Evals Course
date: "2025-07-31"
author: Vishal Bakshi
description: In this blog post, I'm going highlight ideas that stood out to me from the third lesson (Error Analysis, cont'd.) and Chapter 4 (Collaborative Evaluation Practices) of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 3: Error Analysis (cont'd.)

**Idea 1: Open code at the trace-level**. Many errors only emerge in the context of the whole trace and mistakes often cascade across turns (i.e. label only the _first_ failure).

**Idea 2: Manually classify open codes using AI-generated axial codes.** This verifies whether the AI-generated codes are applicable (accurate and relevant). During the lesson ChatGPT missed "Did not invoke tool" in its initial axial code generation based on open codes. This type of mistake is common and is why we manually apply LLM-generated axial codes.

**Idea 3: If possible, reproduce a multi-turn error with a simpler single-turn test case**. For example if a multi-turn conversation fails when the LLM tries to retrieve some information, (e.g. return the correct price for product X), create a new single-turn conversation targeting just that task. This "minimal reproducible error" is analogous to software engineering's "minimal reproducible bug". In both cases, you're cutting through the noise and targeting the single task that fails; this help find the root cause.

## Chapter 4: Collaborative Evaluation Practices

**Idea A: When possible, have only one person make the final judgment call on AI evaluation**.  You want a single person making decisions about the success or failure of the AI outputs to reduce noisy cooks in the kitchen. You need that person to have deep domain knowledge or be someone who represents the target users. Hamel and Shreya call this person the Principal Domain Expert.

**Idea B: Annotator disagreements inform rubric improvements, not retroactive label updates**. You're not trying to win an argument. This idea reminds me of being on an interview panel: when you discuss rubric scores with the highest disagreement, the goal is not for panelists to change their scores, the goal is to come to a common understanding of what rubric item was measuring, and if needed, update your understanding to reach concensus for future candidate assessments. If your process is flawed, try to correct it as soon as you can.

**Idea C: Battle-test your artifacts manually**. The iteratively improved human annotator rubric and the concensus labeled dataset becomes the gold standard used for automated evaluators. The rubric becomes the specification passed to the LLM-Judge. A recurring theme in this course: don't build something in the abstract, build it while grounded in real data. Just as we don't predefine axial codes for failure modes before we look at the data and document open codes, we don't predefine a rubric for the LLM Judge before we look at the data. There's a feedback loop between error analysis artifacts (open codes, axial codes, annotator rubrics, annotation scores) and looking at your data. Looking at data builds your intuition which then informs the error analysis artifacts.

**Idea D: Even with multiple annotators, there's an escalation path.** A benevolent dictator may need to intervene and make the final call if the annotators can't come to concensus. This underscores the importance of identifying a single Principal Domain Expert _even during collaborative evaluation_.