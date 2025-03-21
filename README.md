# bbdc25


## Notes:
- the data is generated 
    -> can we re-verse engineer the data generation process?
        -> or: what are the tytpical features considered for modelling this?
        -> additionally: there must be some variable, even if it is whitheld that informs if a transaction is fraudulent or not -> what would i use if i where to generate such a dataset?
    -> is this based on real world distributions or did they make them up themselves? if the latter, on what basis? is there open research to consider?

- 15% fraudsters -> any model that we can use thresholding with should have an according threshold
    -> potentially the organisers used different percentages for train/val/test 

- there are examples of this kind of task:
    - https://www.kaggle.com/datasets/ealaxi/paysim1 -> this looks very similar from a datastand point -> i would assume the bbdc also used https://changefinancial.com/paysim/ to generate the data
       -> there also are example models, we should try those first
            -> https://www.kaggle.com/code/salmarashwan/building-a-fraud-detection-model-paysim-case-study this has really good results just with a rf, not sure if this applies to this dataset as well, but we'll see. double check cross_validation and features used

- consider model options:
    - transaction -> fraud and then aggregation (current pipeline)
    - aggregated transactions per account -> fraud

- consider using autosklearn or autopytorch, since the data is tabular

- the documentation suggests there are no payments from merchants to customers (ie refunds) -> q auick search in the transactions also suggest, but does not confirm this

-> new aggregation method idea: use predict_proba of model select a subset of transactions (not sure if sorted by highest or similar, or just pad) and then use another model to predict if the account is fraudulent or not => not really better than the current method 

Added new ssl and fraud classification methods, core idea is this: incorporate network info into each transaction, then group the transactions per account and classify directly if those are fraudulent or not -> should scale well (ie not per transaction but also not per dataset). However, it is also key, that we find all of the network info, because it is not in the transactions of a single account.

Realized, that fraudsters do seldom only use cash (2-7 out of 11k accounts) and that we might be able to aggregate the cash transactions in order to condense our data.

Question: does the simulation consider the order of transactions relevant? -> likely not, as it is agent based with a certin prob to perform such actions -> we could aggregate the transactions. If we assume only topological fraud (ie circle, fan-in/out etc) we could even aggregate per account pair. If we assume single transactions as fradulent we would need to be careful in agregation though -> from what i gather the core idea in amlsim (and paysim) is to only flag certain actions during simulation that are creating one of the known topologies.

placative, but might help in shifting perspective: we are given the vertices of a graph that is constructed by the properties of the nodes. We then want to get back to the properties, more specifically the fraudster property.

also: consider the other direction: which accounts are we sure of that are normal people?
- accounts only using cash_in or cash_out (see plot). 
- anything else?

currently working on account lvl classification. will add the exact number of fraudsters to the dataset/model (since we can count them in train, val and engineered it for test)

## Simulations:
- https://bth.diva-portal.org/smash/get/diva2:955852/FULLTEXT06.pdf
- https://github.com/IBM/AMLSim

## Features and Sophisticated Models:
- https://snapml.readthedocs.io/en/v1.15/graph_preprocessor.html
    - https://dl.acm.org/doi/pdf/10.1145/3677052.3698674
    - https://github.com/IBM/snapml-examples/blob/main/examples/graph_feature_preprocessor/graph_feature_preprocessor.ipynb
- https://github.com/IBM/Multi-GNN
    -> python main.py --data Small_HI --model pna --emlps

## Understandings:
- The fraudulent percentage differs between the datasets (train is mostly between 10-15% per action, val ranges from 3-15% per action, test is obviously unclear)
- the aggregation per accountid is based on quantiles atm -> the model before should be tuned for fraud precision over recall as faulty transaction labeling is more costly than missing a fraudulent transaction, specifically as not all transactions of a fraudulent account are necissarily fraudulent
    -> tuning for precision can achieve 100% fraud transaction precision, but recall is low resulting in bad fraudster f1 -> tuning is not worth it atm
    -> adjusted cv split for less agressive optimization and adjusted aggregation method -> results are on par without precision tuning -> might consider to revisit in final models, but not atm
- Ver01 feature set is better (over raw) in rf and brf: roughly doubles fraud precision (transaction) and is ~8 points better in macro f1, 20 in fraud f1 (account)
- Ver02 feature set is better (over v1)
    - brf: 
        - transaction: 2x fraud precision, 1.5x recall (ironically f1 almost identical)
        - account: +20 points fraud f1, +10 points macro f1
    - rf 
        - transaction: fraud: 1.5x precision, 4x recall, 4x f1
        - account: fraud f1: -3 points macro: identical (weirdly only 8% got selected -> because only one of their transactions got flagged and a lower threshold isn't possible)
    => next: either consider better selection/aggregation method or try autopytorch or try the simulation
- RF with 1000 trees instead of 100 is better in transactional (+1 f1, +4pre and +1rec), but ironically not worse (-1) in account f1

# Next Steps:
- submit a version with no fraud to reverse engineer the fraudster percentage
- consider featueres -> stacking, embedding?
- consider cnn model for transactional data, unsure how to train on basically only one sample / how to split the data into multiple samples instead of a single big one -> approach would be nice as it could cross-reference / -calc transactions making the feature space less crucial
- double check if kaggle data is useful or harmful -> train on train+kaggle and validate on val 


## Log:
- read desrciption and added some thoughts
- plot data
- setup pipeline and implemented dummy, rf and brf
- submitted two submissions with rf on train.csv and rf on train+val+kaggle
- added features (mostly from the diss/chatgpt: transactional, some behavioural, some social -> could be more, no agent based)
- re-ran models with v01 features -> pretty good results (see above)
- added precision optimized rf -> results are not better but have higher risk
- realized that the fraudster percentage is more like 13% (at least in train and val) -> does lead to worse results -> probably tune at the end (as f1 is harmonic mean between precision and recall)
- added features v02
- re-ran models with v02 features -> pretty good / on par results (see above)
- investigated aggregation strategies: while interesting, the main source of error remains uncertainty in the transactional model
- uploaded an unchecked submission with rf on train+val+kaggle -> results we're not good, as the model predicted 100% fraud for some reason -> but gave me the idea to reverse engineer the fraud/non-fraud ratio. Would need to submit a version with no fraud though. -> missed that i could have already calculated it here. in the test set we have a precision of 0.114 if all are labeled fraudster -> so we have 11.4% fraudsters in the test set

