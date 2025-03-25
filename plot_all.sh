python3 plot.py --transactions_path ./task/train_set/x_train.ver00.parquet --fraud_path ./task/train_set/y_train.ver00.parquet --val_name train
python3 plot.py --transactions_path ./task/val_set/x_val.ver00.parquet --fraud_path ./task/val_set/y_val.ver00.parquet --val_name val
# python3 plot.py --transactions_path ./task/kaggle_set/x_kaggle.ver00.parquet --fraud_path ./task/kaggle_set/y_kaggle.ver00.parquet --val_name kaggle
python3 plot.py --transactions_path ./task/test_set/x_test.ver00.parquet --fraud_path "" --val_name test
