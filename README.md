

# KGPC

code for "Few-shot biomedical named entity recognition via knowledge-guided instance generation and prompt contrastive learning"
###  Examples Instructions
(1) Train BioNLP11EPI on train dataset, evaluate on dev dataset:
```
python ./BioNER.py train --config configs/train.conf
```

(2) Evaluate the BioNLP11EPI model on test dataset:
```
python ./BioNER.py eval --config configs/eval.conf
```

### Fetch data
datasets lies in data file
