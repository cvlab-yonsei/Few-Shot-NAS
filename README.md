# starlab1


## Get started
```bash
cd mobilenet
mkdir SuperNet/checkpoint Search/checkpoint Evaluation/checkpoint
mkdir -p tb_logs/supernet tb_logs/retrain
```

### Train SuperNet
```bash
bash SuperNet/scripts/train.sh {YOUR_TAG} {YOUR_SEED}
```

### Evol. Search
```bash
bash Search/scripts/evol_search.sh {YOUR_SEED}
```

### Retrain a searched network
```bash
bash Evaluation/scripts/train.sh {YOUR_TAG} {YOUR_SEED}
```
