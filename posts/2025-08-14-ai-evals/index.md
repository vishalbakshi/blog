---
title: Standout Ideas from Lessons 8 of the AI Evals Course
date: "2025-08-14"
author: Vishal Bakshi
description: In this blog post, I highlight standout ideas from the eighth and final lesson of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 8: Course Review and Live Coding an Annotation App

**Idea 1: You should have AI-in-the-loop,** as opposed to being the "human-in-the-loop". You should drive the AI. You should understand every step of the AI pipeline (as is learned through the process of doing error analysis). As we've seen throughout the course, there are strategic opportunities to use LLMs to _supplement_ your analysis, potentially evolving to full automation with routine human validation (e.g. production LLM Judge outputs sampled and reviewed every week).

**Idea 2: Provide a detailed prompt when vibe coding an annotation app.** Shreya provided the direct path to the trace CSV, an explanation of the CSV and message column structures (content/roles), and the goals/key characteristics of the app and its UI/UX (such as open coding, a progress bar, navigation buttons and a dropdown of previously used annotations). Shreya asks the LLM to create a `plan.md` when coding something for the first time and reviews/provides feedback before finalizing it. You can always follow up with more requirements as you review the plan before executing on it.

**Idea 3: Implement heuristics for where the user's attention should go.** One example is semantic and/or keyword highlighting in the displayed trace based on previous annotations. This will help the user more easily identify common failure modes (which supports the core goal: reduce friction! Ease cognitive load!).

**Idea 4: Common preferred annotation app UI characteristics** <mark>(note: these are just examples, you should think about your own app requirements when building it)</mark>: expanding/collapse the system prompt, "keywords from user query" at the top which are highlighted in the messages listed below it, annotation should always be visible (shouldn't require scrolling to view it), highlight domain-specific details (like dates/times for scheduling requests), visually flagged duplicate assistant messages, cleanly render raw JSON messages/tool calls.

**Idea 5: Think step by step.** That means you too, not just the LLM! To improve accuracy, start with low hanging fruit (disambiguate your prompt/instructions), then tackle more involved tasks (decomposing the task into smaller subtasks that are easier for the LLM to handle), and finally approach advanced strategies (e.g. fine-tuning, prompt optimization, human review loops to generate more labeled examples).