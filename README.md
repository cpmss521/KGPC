

# KGPC

code for "Few-shot biomedical named entity recognition via knowledge-guided instance generation and prompt contrastive learning"
###  Examples Instructions
(1) Train NCBI on train dataset, evaluate on dev dataset:
```
python ./main.py train --config configs/train_aug.conf
```

(2) Evaluate the model on test dataset:
```
python ./main.py eval --config configs/eval.conf
```

### Fetch data
datasets lies in data file
