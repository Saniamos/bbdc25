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
    - sequence modelling?

- consider using autosklearn or autopytorch, since the data is tabular

- the documentation suggests there are no payments from merchants to customers (ie refunds) -> q auick search in the transactions also suggest, but does not confirm this

## Understandings:
- The fraudulent percentage differs between the datasets (train is mostly between 10-15% per action, val ranges from 3-15% per action, test is obviously unclear)
- the aggregation per accountid is based on quantiles atm -> the model before should be tuned for fraud precision over recall as faulty transaction labeling is more costly than missing a fraudulent transaction, specifically as not all transactions of a fraudulent account are necissarily fraudulent
    -> tuning for precision can achieve 100% fraud transaction precision, but recall is low resulting in bad fraudster f1 -> tuning is not worth it atm
    -> adjusted cv split for less agressive optimization and adjusted aggregation method -> results are on par without precision tuning -> might consider to revisit in final models, but not atm
- Ver01 feature set is better (over raw) in rf and brf: roughly doubles fraud precision (transaction) and is ~8 points better in macro f1, 20 in fraud f1 (account)
- Ver02 feature set is 

## Log:
- read desrciption and added some thoughts
- plot data
- setup pipeline and implemented dummy, rf and brf
- submitted two submissions with rf on train.csv and rf on train+val+kaggle
- added features (mostly from the diss/chatgpt: transactional, some behavioural, some social -> could be more, no agent based)
- re-ran models with v01 features -> pretty good results (see above)
- added precision optimized rf -> results are not better but have higher risk
- realized that the fraudster percentage is more like 13% (at least in train and val) -> does lead to worse results -> probably tune at the end
- added features v02
- re-ran models with v02 features -> pretty good results (see above)

