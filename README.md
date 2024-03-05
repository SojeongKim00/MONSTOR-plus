# Inductive Influence Estimation and Maximization over Unseen Social Networks under Two Diffusion Models
This is the source code of **MONSTOR+**, *an extended version of the inductive machine learning method, MONSTOR(Ko et al., 2020)* for estimating the influence of given seed nodes in social networks with GNN.
MONSTOR is the first inductive method for estimating the influence of given nodes by replacing repeated MC simulations, however, it only can be used under the IC model. 
Therefore, we propose MONSTOR+ to extend MONSTOR in 2 aspects: **being applicable under two diffusion models, the IC and LT models and improving the performance**. MONSTOR+ contains 2 additional contributions, (a) the augmented node feature and (b) the advanced aggregator.

These codes are based on the code of MONSTOR, and its github link is as below:
<https://github.com/jihoonko/asonam20-monstor>

## Dataset (Graphs)
We used three real-world social networks: Extended, WannaCry, and Celebrity under the IC model with three activation probabilities BT, JI, and LP and the LT model.
| | \|V\| | \|E\| |
|------|---|---|
|Extended|11,409|58,972|
|WannaCry|35,627|169,419|
|Celebrity|15,184|56,538|

## Generating train data
We generate and preprocess the train data with repeated Monte Carlo simulation under the IC and LT models.
```
./compile.sh
./monte_carlo_[IC|LT]_[random|degree] graphs/[Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP].txt
```
```
python processing.py [Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP]
```

## Training
In training, we train the model using two of the three networks for inductive setting. "--target" means the masking networks. For example, if the target is 'Extended', train the model with the rest of the two networks('Celebrity' and 'WannaCry').
```
python train.py --target=[Extended|Celebrity|WannaCry] --input-dim=4 --hidden-dim=32 --gpu=0 --layer-num=3 --epochs=100
```
## Experiment
__IE.py__ : How accurately does MONSTOR+ estimate the influence of seed sets? (Influence Estimation) 

__submodularity.py__ : Is MONSTOR+ submoudular as the ground-truth influence function is?

__IM.py__ : How accurate are simulation-based IM algorithms equipped with MONSTOR+, compared to state-of-the-art competitors? (Influence Maximization)

__scalability.py__ : How rapidly does the estimation time grow as the size of the input graph increase?

```
python [IE|IM|submodularity].py --input-dim=4 --hidden-dim=32 --layer-num=3 --gpu=0 --checkpoint-path=[path_of_target_checkpoint] --prob=[BT|JI|LP] --n-stacks=[number_of_stacks]
```
Before showing the scalability, we obtain the cycle information as an augmented node feature with MATLAB (aug_feat.m).
```
python scalability.py --graph-path=graphs/scal_[20|21|22|23|24|].txt --input-dim=4 --hidden-dim=16 --gpu=0 --layer-num=3 --checkpoint-path=[path_of_target_checkpoint]
```

## The average influence of repeated MC simulations
```
./compile.sh
./test_[IC|LT] [path_of_target_graph] [path_of_seeds]
```
