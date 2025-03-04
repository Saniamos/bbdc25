python3 plot.py --transactions_path ./task/train_set/x_train.parquet --fraud_path ./task/train_set/y_train.parquet --val_name train
python3 plot.py --transactions_path ./task/val_set/x_val.parquet --fraud_path ./task/val_set/y_val.parquet --val_name val
python3 plot.py --transactions_path ./task/kaggle_set/x_kaggle.parquet --fraud_path ./task/kaggle_set/y_kaggle.parquet --val_name kaggle
python3 plot.py --transactions_path ./task/test_set/x_test.parquet --fraud_path "" --val_name test
