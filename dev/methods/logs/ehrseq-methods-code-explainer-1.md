I'll create a comprehensive document addressing all three questions with practical examples and exercises.

[1 tool called]

Perfect! I've created a comprehensive Q&A document at `/Users/pleiadian53/work/ehr-sequencing/dev/methods/temporal_med2vec_qa.md`. Let me summarize the key insights:

## Question 1: Vocab Size in Medical Coding

**Key Insights:**
- **Real-world range:** Typically 5,000-50,000 after filtering (not the full 70k+ codes in ICD-10)
- **Memory impact:** vocab_size=50k, embed_dim=128 → ~51 MB for embeddings
- **Critical practice:** Filter rare codes (min_frequency=10) to reduce vocabulary by 80%+ while covering 95% of occurrences
- **Mixed systems:** Need unified vocabulary when combining ICD-10 + LOINC + RxNorm codes

**Practical recommendation:**
```python
# Filter rare codes first
code_counts = Counter(all_codes)
frequent_codes = {code for code, count in code_counts.items() if count >= 10}
vocab_size = len(frequent_codes)  # Typically reduces by 80%
```

## Question 2: Einsum Mastery

**The TemporalMed2Vec line decoded:**
```python
scores = torch.einsum('be,bce->bc', target_embed, context_embed)
#                     'be,bce->bc'
#                      ││ │││  ││
#                      ││ │││  └┴─ output: [batch, context]
#                      ││ └┴┴───── context: [batch, context, embed]
#                      └┴──────── target: [batch, embed]
#                     'e' repeated → dot product over embedding dimension
```

**What it computes:** Similarity score between each target code and its context codes via batched dot products.

I included extensive practice exercises and a cheat sheet for common operations!

## Question 3: Models and Loss

**Answer: No, models don't always return loss**

**Best practice pattern:**
```python
# Model returns representations
class Model(nn.Module):
    def forward(self, x):
        return self.network(x)  # Returns outputs, not loss

# Loss is computed separately
criterion = nn.CrossEntropyLoss()
loss = criterion(model(x), y)
```

**Why separate?**
- Flexibility: Different losses for different tasks
- Reusability: Model works for both training and inference
- Standard PyTorch pattern (used by torchvision, transformers)

The original TemporalMed2Vec returns loss for tutorial simplicity, but I provided a refactored production version with proper separation of concerns.