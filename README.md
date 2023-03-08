# starlab1


cd mobilenet
mkdir SuperNet/checkpoint
mkdir Search/checkpoint
mkdir Evaluation/checkpoint

## Train SuperNet
bash SuperNet/scripts/train.sh {YOUR_TAG} {YOUR_SEED}

## Evol. Search
bash Search/scripts/evol_search.sh {YOUR_SEED}

## Retrain a searched network
bash Evaluation/scripts/train.sh {YOUR_TAG} {YOUR_SEED}
