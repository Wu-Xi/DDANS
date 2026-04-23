# DDANS
DDANS: Differentiable Dual Anchor Negative Sampling for Graph-based Recommendation

This is our PyTorch implementation for the paper.


## Environment Requirement

The code has been tested running under Python 3.8.0 and torch 2.0.0.

The required packages are as follows:

- pytorch == 2.0.0
- numpy == 1.22.4
- scipy == 1.10.1
- sklearn == 1.1.3
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `tau`
  - It controls the temperature in Gumbel-Softmax, determining how sharp or smooth the sampling distribution is.
- `n_negs`
  - It specifies the size of negative candidate set when using DDANS.

#### LightGCN_DDANS

```
python main.py --dataset ali --dim 64 --lr 0.001 --l2 0.001 --batch_size 2048 --gpu_id 1  --pool mean --ns ddans --tau 1.9 --n_negs 32 > ddans_lightgcn_ali.log

python main.py --dataset yelp2018 --dim 64 --lr 0.001 --l2 0.001 --batch_size 2048 --gpu_id 1  --pool mean --ns ddans --alpha 2 --n_negs 64 > ddans_lightgcn_yelp2018.log

python main.py --dataset amazon --dim 64 --lr 0.001 --l2 0.001 --batch_size 2048 --gpu_id 1  --pool mean --ns ddans --alpha 1.1 --n_negs 64 > ddans_lightgcn_amazon.log
```
