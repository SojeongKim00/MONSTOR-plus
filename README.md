# Inductive Influence Estimation and Maximization over Unseen Social Networks under Two Diffusion Models
This is the source code of MONSTOR+, an extended version of the inductive machine learning method, MONSTOR(Ko et al., 2020) for estimating the influence of given seed nodes in social networks with GNN.
MONSTOR is the first inductive method for estimating the influence of given nodes by replacing repeated MC simulations, however, it only can be used under the IC model.
Therefore, we propose MONSTOR+ to extend MONSTOR in 2 aspects: improving performance and being applicable under two diffusion models, the IC and LT models.

MONSTOR github link : <https://github.com/jihoonko/asonam20-monstor>

## Dataset (Graphs)
We used three real-world social networks: Extended, WannaCry, and Celebrity under the LT model and IC model with three activation probabilities BT, JI, and LP.
| | \|V\| | \|E\| |
|------|---|---|
|Extended|11,409|58,972|
|WannaCry|35,627|169,419|
|Celebrity|15,184|56,538|

## Generating train data
We generate and preprocess the train data with monte carlo simulation under the IC and LT models.

    ./compile.sh
    ./monte_carlo_[IC|LT]_[random|degree] graphs/[Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP].txt
    python processing.py [Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP]

## Training
