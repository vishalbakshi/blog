
---
title: TIL&#58; How LLM-Foundry Ignores Prompt Tokens in `labels` for Instruction Finetuning. 
date: "2025-07-31"
author: Vishal Bakshi
description: Claude's explanation of how LLM-Foundry replaces prompt tokens with `-100` (common loss ignore index) in an instruction-finetuning data batch's `labels` so that loss is not calculated on the prompt tokens.
filters:
   - lightbox
lightbox: auto
categories:
    - LLM-Foundry
---

I recently realized (after a conversation with a colleague) that I didn't know how LLM-Foundry replaces prompt tokens with `-100` to exclude them from the loss calculation. I fed Claude project knowledge the LLM-Foundry GitHub repo and asked it to walk me through this process line-by-line. Since "Chat using integrations can't be shared" I'm pasting Claude's walkthrough here (with a couple of my notes). All the referenced functions are inside [`llmfoundry/data/finetuning/collator.py`](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/finetuning/collator.py)

## Setup: Input Example

Let's say we have this raw training example:

```
{
    'prompt': 'What is photosynthesis?',
    'response': 'Photosynthesis is the process by which plants convert sunlight into energy.'
}
```

After tokenization, this becomes:

```
example = {
    'turns': [
        {
            'input_ids': [1127, 318, 2825, 43943, 30],      # "What is photosynthesis?"
            'labels': [21197, 43943, 318, 262, 1429, 416, 543, 6134, 10385, 4252, 1657, 656, 2568, 13]  # "Photosynthesis is the process..."
        }
    ]
}
```

## Line-by-Line Processing

### Collator Entry Point (`Seq2SeqFinetuningCollator.__call__`)

```
def __call__(self, examples: list[TokenizedExample]) -> dict[str, torch.Tensor]:
    # examples is a list of TokenizedExample objects (batch)
    
    if self.decoder_only_format:
        batch = self._process_and_batch_decoder_only(examples)  # Most common path
```

### Batch Processing (`_process_and_batch_decoder_only`)

```
def _process_and_batch_decoder_only(self, examples: list[TokenizedExample]):
    processed_examples = []
    
    # Step 1: Process each example in the batch through stitch_turns_decoder_only
    input_ids_and_labels = [
        stitch_turns_decoder_only(
            example_turns=example['turns'],           # The prompt/response turns
            target_prompts=self.target_prompts,      # 'none' 
            target_responses=self.target_responses,   # 'last'
            eos_token_id=self.tokenizer.eos_token_id,
        ) for example in examples
    ]
```

### Core Processing (`stitch_turns_decoder_only`)

This is where the magic happens:

```
def stitch_turns_decoder_only(example_turns, target_prompts, target_responses, eos_token_id):
    target_prompts = target_prompts.lower()  # 'none'
    target_responses = target_responses.lower()  # 'last'
    
    # Line 1: Look up the target policy functions
    prompt_to_target = _TARGET_POLICY_LOOKUP[target_prompts]  # Gets _sequence_to_labels_none
    response_to_target = _TARGET_POLICY_LOOKUP[target_responses]  # Gets _sequence_to_labels_last
    
    # Line 2: Initialize output sequences
    input_ids = []
    labels = []
    
    # Line 3: Process each turn in the conversation
    for idx, turn in enumerate(example_turns):
        is_last_turn = idx + 1 == len(example_turns)  # True for our single-turn example
        
        # Line 4: Extract the prompt and response token sequences
        context = ensure_list(turn['input_ids'])   # [1127, 318, 2825, 43943, 30] - "What is photosynthesis?"  
        target = ensure_list(turn['labels'])       # [21197, 43943, 318, ...] - "Photosynthesis is..."
        
        # Line 5: Add EOS token to final response if needed
        if is_last_turn and eos_token_id is not None:
            if target[-1] != eos_token_id:
                target = target + [eos_token_id]   # Append <|endoftext|> token
        
        # Line 6: Build the concatenated input_ids sequence
        input_ids += context  # Add prompt tokens: [1127, 318, 2825, 43943, 30]
        input_ids += target   # Add response tokens: [1127, 318, 2825, 43943, 30, 21197, 43943, 318, ...]
        
        # Line 7: Build the labels sequence using target policies
        labels += prompt_to_target(context, is_last_turn, prompt_cutoff)  # Calls _sequence_to_labels_none
        labels += response_to_target(target, is_last_turn)                # Calls _sequence_to_labels_last
```

> Vishal note: in Line 7, `labels` starts out as an empty list `[]` so the prompt is added first (via `+= prompt_to_target(...)`), after which the response is added (via `+= response_to_target`).

### Target Policy Functions

```
def _sequence_to_labels_none(sequence, is_last_turn, cutoff):
    # Line 1: Convert ALL prompt tokens to -100 (ignore during loss)
    return [CROSS_ENTROPY_IGNORE_INDEX] * len(sequence)  # [-100, -100, -100, -100, -100]

def _sequence_to_labels_last(sequence, is_last_turn, cutoff):  
    # Line 1: Check if this is the final turn
    if is_last_turn:  # True in our case
        return sequence  # Keep actual token IDs: [21197, 43943, 318, ...]
    else:
        return [CROSS_ENTROPY_IGNORE_INDEX] * len(sequence)  # Would be [-100, -100, ...] for non-final turns
```

### Result After `stitch_turns_decoder_only`

```
# Returned values:
input_ids = [1127, 318, 2825, 43943, 30, 21197, 43943, 318, 262, 1429, 416, 543, 6134, 10385, 4252, 1657, 656, 2568, 13, 50256]
#            |----------prompt----------|  |-------------------response---------------------------|  |EOS|

labels =    [-100, -100, -100, -100, -100, 21197, 43943, 318, 262, 1429, 416, 543, 6134, 10385, 4252, 1657, 656, 2568, 13, 50256]
#            |------ignore prompt-------|  |-------------------keep response--------------------|  |EOS|
```

> Vishal note: we added `prompt_to_target` first to the empty `labels` list so the `-100`s come first.


### Back to Batch Processing (continued)

```
# Step 2: Handle sequence length limits
for input_ids, labels in input_ids_and_labels:
    orig_size = len(input_ids)
    
    # Truncate if too long
    if orig_size > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)  # All 1s for actual tokens
    
    # Step 3: Pad sequences to max_seq_len
    n_total = len(input_ids)
    i_pad = [CROSS_ENTROPY_IGNORE_INDEX] * (max_seq_len - n_total)  # Padding with -100
    
    if self.tokenizer.padding_side == 'left':
        labels = i_pad + labels  # Left pad labels with -100
    else:
        labels = labels + i_pad  # Right pad labels with -100 (more common)
    
    processed_example = {
        'input_ids': input_ids,      # Will be padded by tokenizer.pad()
        'labels': labels,            # Already padded with -100
        'attention_mask': attention_mask,  # Will be padded by tokenizer.pad()
    }
    processed_examples.append(processed_example)

# Step 4: Final padding using HuggingFace tokenizer
batch = self.tokenizer.pad(
    processed_examples,
    padding='max_length',
    max_length=max_seq_len,
    return_tensors='pt',
)
```

### Final Batch Output

```
batch = {
    'input_ids': torch.tensor([
        [1127, 318, 2825, 43943, 30, 21197, 43943, 318, ..., 0, 0, 0]  # Padded with pad_token_id (0)
    ]),
    'labels': torch.tensor([
        [-100, -100, -100, -100, -100, 21197, 43943, 318, ..., -100, -100, -100]  # Padded with -100
    ]),
    'attention_mask': torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, ..., 0, 0, 0]  # 1 for real tokens, 0 for padding
    ])
}
```

### Key Points

1. Prompt tokens → Labels = -100: The _sequence_to_labels_none function converts all prompt tokens to -100
2. Response tokens → Labels = actual token IDs: The _sequence_to_labels_last function keeps response tokens as-is
3. Padding tokens → Labels = -100: All padding positions also get -100
4. Loss calculation: PyTorch's CrossEntropyLoss(ignore_index=-100) will only compute loss on response tokens

This ensures the model learns to generate responses but never tries to "predict" the input prompt!
