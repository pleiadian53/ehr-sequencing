@ehr-sequencing/src/ehrsequencing/models/lora.py:19-24 



You previously had documented LoRA under examples/pretrain_finetune/ 



But this layer is still somewhat myterious to me. I understand it'll inject a lower-rank matrix B by A in order to reduce the trainable parameters. Does it always freeze the original weight matrix W first? 



Where can LoRA apply to? I guess anywhere with parameter matrix like W, such as the weight matrices associated with Q, K, and V in the transformer architecture, and Wo, the output weight matrix? 



Does it come after layer normalization or are they related at all? What is it used in PyTorch? 

---

@ehr-sequencing/src/ehrsequencing/models/lora.py:19-24 



You previously had documented LoRA under examples/pretrain_finetune/ 

But this layer is still somewhat myterious to me. I understand it'll inject a lower-rank matrix B by A in order to reduce the trainable parameters. Does it always freeze the original weight matrix W first? 

Where can LoRA apply to? I guess anywhere with parameter matrix like W, such as the weight matrices associated with Q, K, and V in the transformer architecture, and Wo, the output weight matrix? 

Does it come after layer normalization or is their ordering a concern at all? Given a foundation model, either custom models or downloaded from Huggingfase, how are they generally used with LoRA in PyTorch? 