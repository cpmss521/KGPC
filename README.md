

# KGPC

Code for "Few-shot biomedical named entity recognition via knowledge-guided instance generation and prompt contrastive learning"
###  Examples Instructions
(1) Train the model on k-shot support set, evaluate on dev dataset:
```
python ./main.py train --config configs/train_aug.conf
```

(2) Evaluate the model on query set:
```
python ./main.py eval --config configs/eval.conf
```

Simulate low-resource scenarios by down-sampling:
```
python  sample_k_shot.py
```
