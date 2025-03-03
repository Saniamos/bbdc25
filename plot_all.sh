python3 plot.py --transactions_path ./task/train_set/x_train.csv --fraud_path ./task/train_set/y_train.csv --val_name train
python3 plot.py --transactions_path ./task/val_set/x_val.csv --fraud_path ./task/val_set/y_val.csv --val_name val
python3 plot.py --transactions_path ./task/kaggle_set/x_kaggle.csv --fraud_path ./task/kaggle_set/y_kaggle.csv --val_name kaggle
python3 plot.py --transactions_path ./task/test_set/x_test.csv --fraud_path "" --val_name test
