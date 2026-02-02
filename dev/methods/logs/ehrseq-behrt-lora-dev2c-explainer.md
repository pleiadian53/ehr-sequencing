In this session, let's focus on the EHR-sequencing project. 



First, can you walk me through examples/pretrain_finetune/train_behrt_demo.py, which depends on models defined under src/ehrsequencing/models/ 



In particular, I'd like to focus on learning the model design, definition and usage such as @ehr-sequencing/src/ehrsequencing/models/behrt.py. For example, does it use a pretrain model from huggingface? How do you provide pretrained embeddings (e.g. from Med2Vec) and how to do you apply LoRA to a foundation model, etc.

Please document this under dev/models/pretrain_finetune/ 

PS: dev/ keeps our private notes, not to be shared on github, as opposed to docs/



---

