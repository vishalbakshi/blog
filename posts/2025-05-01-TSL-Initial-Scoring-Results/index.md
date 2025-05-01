---
title: Initial Manual Scoring Results for TinyStories Models
date: "2025-05-01"
author: Vishal Bakshi
description: A detailed breakdown of my manual evaluation of three TinyStories language models (1M, 8M, and 28M parameters) across six capability categories. I share scoring methodology, surprising findings about emergent reasoning in small models, and comparisons to the original TinyStories paper. This analysis establishes baselines both for my own model training project and my eventual LLM Judge, and reveals how different capabilities scale with model size.
filters:
   - lightbox
lightbox: auto
categories:
    - LLM
    - deep learning
    - TinyScaleLab
---

## Recap

In this post, I'm going to analyze the initial manual scoring results for my baseline models' text generations given my 150 evaluation prompts across six scoring categories and 18 criteria. A quick recap of what I've done so far:

- Defined scoring criteria
- Curated a set of eval prompts based on each scoring category
- Created a fast HTML app where I can perform my scoring activities

## Evaluation Categories

I have six scoring categories that I'm evaluating my models on:

### Foundational language capabilities
- Grammar
- Context-Tracking (Consistency)

### Emergent capabilities
- Factual Knowledge
- Reasoning
- Creativity

### Story-related capabilities
- Plot

My goal was to generate prompts that either isolate (Factual Knowledge, Reasoning, Context-Tracking) or elicit opportunities to exhibit (Plot, Creativity) scoring categories. I wanted to make the job easier first for myself, and then use that as a proxy of making the job of the LLM judge easier to evaluate scoring categories in a focused way.

## Baseline Models

I've chosen three models as my baseline because they're similar in size to the models that I'm going to be training in this project:

- TinyStories-1M (~3.7 million parameters)
- TinyStories-8M (~20 million parameters)
- TinyStories-28M (~60 million parameters)

## Generation Script

I'm using a pretty standard generation script. Things I want to highlight:

- Making sure the padding side is left so that we're not generating tokens based on padding tokens
- `model.eval()` and `torch.no_grad()` are things that I always make sure to do so that it's somewhat deterministic when it's expected to be deterministic
- I'm doing `do_sample=False` and `num_beams=5` because that was published by the authors as their parameters for generation
- I have a minimum and maximum length, which I'll talk about at the end about how I think that might change moving forward

## Eval Prompts

My current eval prompts set includes:

- 25 unique prompts for Reasoning
- 25 unique prompts for Factual Knowledge
- 25 prompts each for Context-Tracking, Plot and Creativity (with some overlap)
- 25 prompts for Grammar (5 prompts sampled from the other 5 categories)

That's 150 total prompts.

## Scoring Methodology

I have six categories across 18 criteria, evaluating generations from three models on 150 prompts each. The scores that I'm providing for each criteria are either 0, 0.5, and 1.0, taken from the Tiny Stories paper (though they didn't quite use it the same way I'm using it), in Section 4.2 (Figures 9/10/11) where they use scoring levels success (green), failure (red), and partial success (yellow).

## Overall Results

First, let's look at the average value across all categories and criteria for each model:

| model_name | score_value |
|------------|-------------|
| roneneneldan/TinyStories-1M | 0.25 |
| roneneneldan/TinyStories-8M | 0.49 |
| roneneneldan/TinyStories-28M | 0.61 |

As I would expect, as model size increases, the average score value increases. The 1M parameter model (which actually has 3.7M parameters) has an average score of 0.25. The 8M parameter model (which is closer to 20M parameters) has an average score of about 0.5. And the 28M parameter model (which has about 60M parameters) has an average score of 0.61.

A parameter count _increase_ of 4x (16.3M increase from 3.7M to 20M) yields an overall mean score _increase_ of 1x (0.25 to 0.50). A parameter count _increase_ of 2x (40M increase from 20M to 60M) yields an overall mean score _increase_ of 25% (0.49 to 0.61). There are decreasing gains overall when increasing parameter count. For a 125M parameter model (that I'm planning to train), I would expect <10% increase from a mean overall score of 0.61.

## Scores by Category

Next, let's look at how these models are doing for each of the categories overall:

| | 1M | 8M | 28M |
|------------------|------|------|------|
| Context-Tracking | 0.14 | 0.51 | 0.63 |
| Creativity | 0.12 | 0.16 | 0.32 |
| Factual Knowledge | 0.08 | 0.32 | 0.40 |
| Grammar | 0.59 | 0.82 | 0.86 |
| Plot | 0.10 | 0.42 | 0.60 |
| Reasoning | 0.20 | 0.44 | 0.70 |

Some interesting things to point out:

The highest category by score for my 1M parameter model is grammar, by farL 0.59. That's about three times as large as any other category. This is in line with what I read in the TinyStories paper, that grammar appears first as a capability. 

The worst categories, even for the largest model that I tested, were Creativity and Factual Knowledge. Creativity in particular was the lowest scoring, and this also tracks with the TinyStories paper, because they had shown that creativity only really appears at large hidden dimension sizes. And even then, the maximum value of creativity (8s and 9s out of 10) was only available for models like GPT-4.

Factual Knowledge was also significantly lower than the other four categories. 

The other category I want to highlight is Reasoning. The Reasoning score for the smallest model is 0.2, it doubles to 0.44 at 8M, and then it goes up by another 60 percent to 0.7 for the 28M model. That's pretty solid! 70%, 7 out of 10. So, if we were talking about school grades, a 70 percent is passing. Very cool to see reasoning potential, even for the tiniest model evaluated.

In every case, there is an increase as we go from 1M to 8M to 28M model name. In some cases, the jump comes later, such as for Creativity. In most cases, the jump happens between the 1M and 8M models. 

## Scoring by Criteria

Now let's look at each criteria for each category:

### Emergent Capabilities: Creativity, Factual Knowledge and Reasoning

| Factual Knowledge | 1M | 8M | 28M |
|------------------------------------------------------------------|------|------|------|
| Completion contains only correct factual information relevant to the prompt | 0.08 | 0.32 | 0.4 |

| Reasoning | 1M | 8M | 28M |
|---------------------------------------------------------------------|------|------|------|
| Completion demonstrates correct logical reasoning relevant to the prompt | 0.2 | 0.44 | 0.7 |

| Creativity | 1M | 8M | 28M |
|---------------------------------------------------|------|------|------|
| Character behavioral and emotional responses are innovative | 0.00 | 0.04 | 0.22 |
| The completion contains unique details to the story world | 0.02 | 0.12 | 0.34 |
| The completion creates fresh situations | 0.00 | 0.08 | 0.20 |
| The completion offers unexpected or novel elements | 0.48 | 0.42 | 0.50 |

Factual Knowledge and Reasoning only had one criteria each. For Factual Knowledge, I was assessing if the completion contains only correct factual information relevant to the prompt. For Reasoning, I was assessing if the completion demonstrates correct logical reasoning relevant to the prompt.

For Creativity, note that the smallest model performs well for the criteria "The completion offers unexpected or novel elements." Since I was isolating Grammar, Plot and Context-Tracking from Creativity, the tiniest model could deviate from Plot/Context and still get a high score for this criterion, making it the lowest bar to cross. For the other three Creativity criteria, the 1M model has negligible skill.


### Foundational Language Capabilities: Grammar and Context-Tracking

| Grammar | 1M | 8M | 28M |
|--------------------------------------------------|------|------|------|
| Age-appropriate vocabulary usage | 1.00 | 1.00 | 0.98 |
| Dialogue formatting and punctuation | 1.00 | 0.96 | 0.98 |
| Proper use of pronouns and referents | 0.56 | 0.88 | 0.90 |
| Sentence structure logic, clarity and completion | 0.14 | 0.62 | 0.70 |
| Tense consistency throughout the completion | 0.26 | 0.66 | 0.74 |

| Context-Tracking | 1M | 8M | 28M |
|-----------------------------------------------------------------------|------|------|------|
| Completion maintains complete coherence with prompt | 0.20 | 0.62 | 0.64 |
| Correctly references/tracks all objects, characters, and their attributes | 0.20 | 0.52 | 0.68 |
| Maintains consistent narrative flow | 0.02 | 0.40 | 0.56 |

For Grammar, the age-appropriate vocabulary usage was the easiest to score. These models don't really generate anything that's not within the TinyStories dataset. 

Sentence structure, logic, clarity, and completion had the biggest jump from 1M to 8M, going from 0.14 to 0.62. That matches my experiencing scoring: the small model had terrible structure, logic, clarity, and completion in its completions. 

For context tracking, I was looking at three criteria. The biggest jump is from 2% to 40% for maintaining a consistent narrative flow. The medium-sized models were definitely not perfect, but was much better at following the narrative flow of the story.

### Story-Related Capabilities: Plot

| Plot | 1M | 8M | 28M |
|------------------------------------------------------------------------------|------|------|------|
| Conflicts are addressed rather than abandoned | 0.00 | 0.42 | 0.60 |
| The pacing is appropriate (not too rushed or dragging) | 0.00 | 0.14 | 0.24 |
| The story has a clear beginning, middle, and end appropriate to age level | 0.26 | 0.50 | 0.72 |
| The story maintains focus on the central conflict/theme without random diversions | 0.12 | 0.64 | 0.84 |

For Plot, I found the pacing to be the worst category across all models. This checks out with my experience as I was grading these stories - I didn't really get a sense that there was a well-paced story. Either it was dragging and repeating itself slightly, or it was just one or two sentences and insufficient.

For "conflicts are addressed" we go from 0% to 42% from 1M to 8M. The smallest model simply ignored or abandoned conflicts that were in the premise and the prompt. The other big jump is for "focusing on the central theme" - the smallest to medium model had almost a 3x jump, and then there was still a considerable 30% jump from the medium to large model. 

## Comparison to TinyStories Paper

I'm going to revisit the targets that I established from Figure 4 of the TinyStories paper, where they showed the different scores based on hidden dimension and number of layers. I matched that up with the three models that I'm testing:

### Creativity

The TinyStories paper reported:
- 1M: 0.47
- 8M: 0.65
- 28M: 0.69

My scores:
- 1M: 0.12
- 8M: 0.16
- 28M: 0.32

This was really interesting - I was expecting my assessment to be maybe a little lenient, but it turns out that's not the case. My scores were significantly lower. The 28M parameter model (which is actually 60M) got 70% for creativity in the paper, while mine was at 30%. I might have to change that criteria over the course of this project, or it might turn out that for creativity, I have a stricter judge.

### Grammar

TinyStories:
- 1M: 0.61
- 8M: 0.77
- 28M: 0.83

My scores:
- 1M: 0.59
- 8M: 0.82
- 28M: 0.86

This matched out pretty well! 61%/59%, 77%/82%, and 83%/86%. The most common baseline capability matches between the targets and my relatively rough evaluation, so thumbs up!

### Context-Tracking (Consistency)

TinyStories:
- 1M: 0.45
- 8M: 0.80
- 28M: 0.90

My scores:
- 1M: 0.14
- 8M: 0.51
- 28M: 0.63

Similar to creativity, it turns out that my criteria or my judging is a lot stricter than the GPT-4 evaluator used in the paper. The highest score in the paper was 90%, whereas mine was 63%. I'm not too worried about this - I would much rather be stricter than not. However, I'll be open to changing my approach later on if that turns out to be a problem.

### Plot

TinyStories:
- 1M: 0.44
- 8M: 0.72
- 28M: 0.73

My scores:
- 1M: 0.10
- 8M: 0.42
- 28M: 0.60

The 28M scores for Plot are in the same range but medium-sized and small model scores are significantly different.

Factual Knowledge and Reasoning were not quantitatively assessed in the TinyStories paper in the way that these other scores were listed, so I don't have those reference points for my evaluation.

## Observations From Manual Scoring

After manually scoring 450 stories, I have some observations:

1. Judging quality improves (and changes) over time
   - Implicit judging criteria surfaces over time.
   - By the time I was doing the last hundred, I realized that I was a lot more definitive in giving 0s, 0.5s, and 1s.
   - Thee largest model likely has the strictest scores (it was graded last).

2. Phrasing of scoring criteria improved
   - I wanted to be able to answer the question as fast as I could (450 stories to get through!) with a quick yes, no, maybe (1, 0, 0.5).
   - Initially, some of the criteria were phrased as questions, requiring more cognitive work. I expect that rephrasing the criteria as statements will also ease the "cognitive load" for my LLM judge.

3. I identified one duplicate prompt and replaced it

4. Pros and cons of isolating scoring categories
   - I scored each category in isolation.
   - More times than not, I found this very liberating---I could assess Creativity without worrying about Context-Tracking or Plot.
   - However, language is very difficult to compartmentalize. If something's not factually correct, it will be a distraction when assessing Reasoning. If the context is not being tracked, it makes it harder to assess plot.
   - Regardless, I thought this isolation of scoring categories overall benefited my approach

## Exciting Discoveries

The main takeaway for me, which was very cool to see, is that Reasoning and Factual Knowledge capabilities exists even for the smallest model. The 1M model scored 20% on Reasoning - that's not nothing! 

The fact that there are non-zero values for these tiny models is really mind-blowing to me. It's really exciting because there's potential. We can do something with this, especially as these are just pre-trained models---we haven't fine-tuned them yet. What can we do with this Reasoning and Factual Knowledge capability? That's what really excites me moving forward.

## Process Improvements

When generating completions for my Reasoning and Factual Knowledge, I want to remove the `min_length` parameter for `model.generate()` because I don't want there to be a minimum generation length when the answer can be a few tokens, forcing the model to uneccesarily elongate the story. However, I won't make this change for my LLM judge as I want to compare its scores with mine for the same prompt/completion pairs.

## Next Steps

With a full eval set scored, I can now move on to prompt engineering an LLM judge (I'll be using Gemini 2.5 Flash and Claude Haiku 3.5). My goal is for a 90%+ alignment between my scores and the LLM judge before I choose to use it for future experiments.

Follow along this project (and others) in my [YouTube channel!](https://www.youtube.com/@vishal_learner).