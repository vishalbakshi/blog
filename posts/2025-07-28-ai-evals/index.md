---
title: Standout Ideas from Lesson 2 of the AI Evals Course
date: "2025-07-28"
author: Vishal Bakshi
description: In this blog post, I'm going highlight ideas that stood out to me from the second lesson (Error Analysis) of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 2: Error Analysis

**Idea 1: Error analysis is how we close the gulf of comprehension.** The gulf of comprehension is the gap between the information the data contains and your understanding of it. While it's either impossible or unreasonable to look at every single piece of data collected from your users, Hamil and Shreya recommend looking at at least 100 traces. 

**Idea 2: You need ~100 diverse traces to get a good representation of failure modes.** A trace is a full record of the pipeline's interaction (initil user query, all LLM inputs/outs, intermediate reasoning or tool calls, and final user-facing result). The number 100 is not a hard and fast rule, but signifies that 10-15 traces is not going to be enough. You want to look at enough traces such that you reach a theoretical saturation of failure modes. What that means is you've seen enough traces that are diverse enough that looking at any more traces will not introduce any new failure modes that you haven't seen yet. 

**Idea 3: Open coding is Hamel's favorite subject in evals** Open codes are brief, descriptive notes about any observed problems, surprising actions or where behavior feels wrong or unexpected. As you write open codes, categories of errors will emerge.

**Idea 4: During open coding, don't find the root cause, just observe and note.** As you come across failed traces, it's tempting to make note of why you think this failure occurred. For example, if you see that the LLM tool call did not contain the correct information, it's tempting to step back and think about why that took place. Maybe the input was incorrect because your voice agent converted speech to text incorrectly. Maybe there's something wrong in the database which resulted in leading to missed matches. As you can see, this type of pontification can potentially be endless and is not fruitful for the focused process of identifying failure modes. Just note the failure and move on to the next trace. We'll think about root causes later.

**Idea 5: Pull the main failure mode from each trace. don't get embroiled in the details.** Relatedly, it's tempting to zoom in as deep as you can, putting each word of the trace under a microscope. For folks who like analyzing data, this is satisfying. But it introduces noise from your high-priority failure modes. These open codes are later going to be clustered by theme (Axial Coding) and higher-level themes will be extracted; too granular of an analysis is a waste of time.  

**Idea 6: The quality of open coding is going to hinge on your product sense.** Successful open coding depends on your ability to skillfully look at data. This involves two broad skills: 1) you have to know what to look for, 2) you have to be detail-oriented enough to find the failure by looking at a sequence of messages. Knowing what to look for requires a deep understanding of your product from the user's perspective. 

**Idea 7: Start with a simple approach to get value fast.** Fatigue from cognitive load is a real thing, especially when you're looking at a hundred traces, many of which can involve multiple messages between the user and the assistant and multiple tool calls. The success of error analysis is going to depend on how fresh your mind is during the process. To achieve this, keep your open coding and axial coding heuristics simple. You can always do a second, third, and fourth pass through future traces after you fix the most pressing failure modes. 

