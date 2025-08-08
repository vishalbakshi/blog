---
title: Standout Ideas from Lessons 5 + 6 and Chapter 7 from the AI Evals Course
date: "2025-08-07"
author: Vishal Bakshi
description: In this blog post, I highlight standout ideas from the fifth and sixth lessons and Chapter 7 (Evaluating Retrieval-Augmented Generation (RAG)) of the AI evals course by Hamel Husain and Shreya Shankar.
filters:
   - lightbox
lightbox: auto
categories:
    - AI Evals
---

## Lesson 5: Architecture-Specific Evaluation Strategies

**Idea 1: Include edge cases in few shot examples** show the LLM examples that you might struggle with, give it a lot of information in each example. I imagine that you will gain a better understanding of what truly contributes to Pass or Fail judgments as you curate difficult few shot examples. Don't just randomly pick examples, cherry pick them based on how well they complement the rest of your prompt. Give examples that are the most instructive.

**Idea 2:Use automation strategically**. We don't want to not look at our data, but we also want to use the reasoning power of LLMs. Shreya fed an LLM her open codes, axial codes and traces and asked it to label true/false (if LLM responses in trace are substantiated with tool outputs) for each trace and provide a rationale. Shreya trusted the LLM's true/false labels because they provided it with open codes.

**Idea 3: Don't provide open/axial codes in LLM Judge Prompt few shot examples.** We're using this judge in production on unlabeled traces which will not have open/axial codes so we don't want the LLM Judge to learn/expect these codes to be present.

**Idea 4: Aim for 80-85% LLM Judge TNR and TPR.** 50% is random chance, 100% probably means something's wrong in your judge prompt.

**Idea 5: The fastest way for you to fail in an AI project is for people to lost trust in what you're doing.** There's a human bias that people have in trusting what a computer says. Don't take the judge at face value. Run some tests to evaluate the confidence interval of the Judge, unbiasing it's success rate (bias = Judge labels "Pass" more than "Fail" by default or vice vesa). The eval should align with the product experience.

## Lesson 6: RAG, CI/CD

**Idea 1 Do error analysis on the whole system, but do evals on retrieval and LLM generation separately:** Make sure your retriever's Recall@k is 80%+, then perform error analysis, otherwise you're evaluating generation errors based on flawed context. Don't use popular metrics to evaluate generation---measure what's relevant to your product, which you will uncover during your analysis.

**Idea 2 Bring domain knowledge to chunk size:** Is there a natural breaking point in your document? What is a meaninfgul chunk in the context of your domain? The chunks ultimately represent the document during search. It's okay to have variable chunk sizes.

**Idea 3 Likert scales have a use!** Shreya asks an LLM to score synthetic queries (when creating an evaluation dataset) on a Likert scale and filters out queries with scores of 1 or 2 (out of 5). These scores are discarded after this filtering use.

**Idea 4 Ground synthetic queries in realism:** User queries are often confusing to interpret, incomplete, and contain typos/grammatical errors. The queries in your evaluation dataset should reflect such nuances to provide meaningful use cases for retrieval.  
 
**Idea 5 Optimize for Recall@k first:** Shreya has rarely seen utility in optimizing for Precision@k first (how many of the top-k retrieved chunks are relevant?) because the consumer of these chunks in a RAG pipeline is an LLM, which cares more about how many of the total relevant chunks are present in the top-k retrieved chunks (Recall@k) to generate a relevant response. LLMs are getting better at reasoning over the retrieved chunks to determine relevance. Use MRR@k (how high up in the ranking is the first relevant chunk?) after optimizing for Recall@k as MRR@k measures how quickly the LLM finds _an_ answer. 

**Idea 6: Focus on process, not tools.** Which goes against how most people think about building AI systems. If something's not working, your first instinct should be to actually understand what is going wrong, not to plug-in a different tool in hopes for improvement, or sweep different hyperparameters. Additionally, don't get lost in the vector DB sauce---start with basic BM25 keyword search first.


## Lesson 7: Evaluating Retrieval-Augmented Generation (RAG)

**Idea 1: Multi-stage retrieval and the Recall/Precision trade-off**. LLMs can handle many passages so we want to make sure it's provided as many relevant passages as possible, this means increasing the number of passages provided. However, long contexts cost more and are limited by the LLMs context window. We use a cheaper retriever to do a first pass on retrieving relevant passages (Recall@k) and a more powerful retriever to then re-rank them (Precision@k, MRR or NDCG@k). We take the top-k (where k is smaller than the first pass retrieved passages) and pass that to the LLM. 

> Modern LLM attend more strongly to salient tokens, so they can often ignore irrelevant content if the key information is present. But if that information is missing altogether--low recal--then the generator has no way to produce a correct answer.

**Idea 2: The ARES framework complements error analysis.** Precisely evaluating "Answer Faithfulness" (failures include hallucinations, omissions and misinterpretations) and "Answer Relevance" (failure = factually correct based on context but irrelevant to the query) requires error analysis to identify where and how specific failure modes occur in our product.

**Idea 3: Be wary of synthentically generated queries** as they often are not representative of the messy queries encountered in production. Regularly validate these queries, referncing real queries from logs or human-curated examples.