---
title: Standout Ideas from Lesson 4 and Chapter 5 of the Course Reader from the AI Evals Course
date: "2025-08-03"
author: Vishal Bakshi
description: In this blog post, I highlight standout ideas from the fourth lesson and Chapter 5 (Automated Evaluators) of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 4: Automated Evaluators

**Idea 1: You can't measure what you don't ask for.** If your prompt didn't include an instruction on providing links to X, and the LLM doesn't provide those links, that's a specification failure (fix the prompt!). If the prompt did include it but the LLM failed to apply the instruction, that's a generalization failure and should be tracked by an automated evaluator. 

**Idea 2: Use code-based evaluators if you can,** as they are deterministic. They take as input the trace and a failure mode and return `True` or `False` or some score for objective rule-based checks (e.g. parsing structure, regex/string matching for keywords, structural constraints, tool execution errors).

**Idea 3: Just because you can ask an LLM Judge anything you want, doesn't mean you should.** Use LLM Judges to do specific, well-defined, binary failure mode classification (Pass/Fail) tasks. A Pass/Fail LLM Judge score is easier to assess and leads to easy-to-interpret Judge accuracy. Also, don't pack multiple criteria into one prompt, create a prompt for each criterion.

**Idea 4: Don't leak test instances into your process of building an LLM Judge** Use 10-20% of labeled axial coding data to curate Judge prompt few shot examples (training set), ~40% to iteratively improve the prompt (dev set), and ~40% for final unbiased Judge evalation after prompt tuning is done (test set). The last thing you want in your prompt is a few shot example that's in the test set. <mark>Low dev set performance (TPR and TNR) tell us that the few shot examples from the train set do not generalize.</mark>

**Idea 5: Don't show your LLM Judge what it's already good at.** Your few shot examples should show difficult/tricky situations for evaluation. To do this, manually iterate the examples. I would imagine that like open coding, you would build a more nuanced intuition about your data (and your product!) through this process. Other tips: write your examples like you are explaining it to a human; try to include the best example of a pass or fail; your examples can also contain reasoning to provide richer "grounding" to your LLM.

## Chapter 5: Implementing Automated Evaluators

**Idea A: Reference-based and reference-free metrics serve different purposes.** A "reference" here means "reference LLM output". Reference-based metrics allow iterative development with holistic checks (whether the LLM output match the golden reference). Reference-free metrics better adapt at scale on new, unlabeled data (as they measure intrinsic properties or rules related to failure modes). Reference-based metric: LLM output matches a "golden" trace with a specific sequence of tool calls. Reference-free metric: LLM output contains valid tool call names.

**Idea B: Test your judge on unlabeled data.** Even the test set is biased as it represents a portion of our labeled data, which may not be representative of broader out-of-domain situations your Judge will inevitably encounter. We use the Judge's "raw success rate" (number of Pass labels/number of unlabeled traces) and a series of calculations on random test set samples to estimate within a confidence interval the Judge's "true success rate."

**Idea C: Judges don’t come pretrained on a product’s values—we have to teach them.** This is why we validate the judge's response and calculate metrics around alignment (like TNR and TPR). This is also why we provide few shot examples so the Judge can evaluate on the specific desired characteristics of our product (especially for vibe-y dimensions like tone).