# Inductive Influence Estimation and Maximization over Unseen Social Networks under Two Diffusion Models
This is the source code of **MONSTOR+**, *an extended version of the inductive machine learning method, MONSTOR(Ko et al., 2020)*. The preliminary version, MONSTOR, is the first inductive method for estimating the influence of given nodes in social networks with GNN by replacing repeated MC simulations under the IC model. We present the extended version of MONSTOR, MONSTOR+, which enhances MONSTOR by incorporating auxiliary structural node features and an advanced pooling function. Both enhancements lead to improvements in accuracy. Furthermore, while MONSTOR is tailored for the IC model, MONSTOR+ is applied to both the IC and LT models.

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
./monte_carlo_[IC|LT]_[random|degree] graphs/[Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP|LT].txt
```
```
python processing.py [Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP|LT]
```

## Training
In training, we train the model using two of the three networks for inductive setting, resulting in three scenarios:

  • E+W: Training using Extended and WannaCry; and test using Celebrity,
  
  • E+C: Training using Extended and Celebrity; and test using WannaCry,
  
  • W+C: Training using WannaCry and Celebrity; and test using Extended.
  
"--target" means the masking networks. For example, if the target is 'Extended', train the model with the rest of the two networks('Celebrity' and 'WannaCry').

```
python train.py --target=[Extended|Celebrity|WannaCry] --input-dim=4 --hidden-dim=32 --gpu=0 --layer-num=3 --epochs=100
```
## Experiment
All the details are in our paper.

__IE.py__ : How accurately does MONSTOR+ estimate the influence of seed sets? (Influence Estimation) 

__IM.py__ : How accurate are simulation-based IM algorithms equipped with MONSTOR+, compared to state-of-the-art competitors? (Influence Maximization)

__submodularity.py__ : Is MONSTOR+ submoudular as the ground-truth influence function is?

__scalability.py__ : How rapidly does the estimation time grow as the size of the input graph increase?

```
python [IE|IM|submodularity].py --input-dim=4 --hidden-dim=32 --layer-num=3 --gpu=0 --checkpoint-path=[path_of_target_checkpoint] --prob=[BT|JI|LP|LT] --n-stacks=[number_of_stacks]
```
For showing the scalability, we use the local cycle counts as auxiliary node features with MATLAB (aug_feat.m).

```
python scalability.py --graph-path=graphs/scal_[20|21|22|23|24|].txt --input-dim=4 --hidden-dim=16 --gpu=0 --layer-num=3 --checkpoint-path=[path_of_target_checkpoint]
```

## The average influence of MC simulations
```
./compile.sh
./test_[IC|LT] [path_of_target_graph] [path_of_seeds]
```
