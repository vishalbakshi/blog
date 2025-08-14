---
title: Standout Ideas from Lesson 7 of the AI Evals Course
date: "2025-08-12"
author: Vishal Bakshi
description: In this blog post, I highlight standout ideas from the seventh lesson of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 7: Interfaces for Human Review

**Idea 1: Custom UIs = 10x review throughput** compared to reviewing in a spreadsheet. This because custom UIs allow a domain-aware view (emails structured like your inbox instead of a string of text) and hotkeys for navigation or one-click tags, _and_ takes only 1 hr to prototype nowadays. A middle-ground between spreadsheets and custom UIs: jupyter notebook (a pseudo-interface) especially with the [`_repr_html_` method](https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display:~:text=_repr_html_%20should%20return%20HTML%20as%20a%20str).

**Idea 2: HCI Principles for UIs (Nielsen, 1994).** Visibility of status (let your user know where they are), recognition over recall (assign tags instead of free-form text in second round of error analysis and beyond), match the real world (native end user display form; results in catching errors only apparent in this form), user control (pass/fail 1-key press, undo, tag select w/number keys, "defer" for uncertainty, goal: _get the user into a flow state_), minimalist first (expand on demand). Add a progress bar whenever you're making a user wait for something. Overall principle: <mark>reduce friction</mark>.

**Idea 3: Nerd-snipe your features.** Shreya implemented a highlight feature where on the backend their app looks for semantic or keyword similarities with previous failed samples and highlights those words in the current example display to flag common issues for easier user identification. Super cool. Another similar example: batch-label similar traces after clustering to wipeout repeat bugs. I never considered integrating machine intelligence into error analysis before this!

**Idea 4: Criteria drift happens!** Reviewers' definitions change over time so keep rubrics and labels editable. What you think was acceptable/unacceptable changes as you review real traces. Additionally, humans' understanding of LLM capabilities also evolves over time (i.e. humans align with LLMs as LLMs align with humans).