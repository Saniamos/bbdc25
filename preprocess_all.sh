VERSION="ver12"

echo '-- Train ---'
time(python3 preprocess.py --input_path ./task_orig/train_set/x_train.csv --output_path ./task/train_set/x_train.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)
time(python3 preprocess.py --input_path ./task_orig/train_set/y_train.csv --output_path ./task/train_set/y_train.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)

echo '-- Val ---'
time(python3 preprocess.py --input_path ./task_orig/val_set/x_val.csv --output_path ./task/val_set/x_val.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)
time(python3 preprocess.py --input_path ./task_orig/val_set/y_val.csv --output_path ./task/val_set/y_val.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)

echo '-- Test ---'
time(python3 preprocess.py --input_path ./task_orig/test_set/x_test.csv --output_path ./task/test_set/x_test.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)

# echo '-- Kaggle ---'
# time(python3 preprocess.py --input_path ./task_orig/kaggle_set/x_kaggle.csv --output_path ./task/kaggle_set/x_kaggle.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)
# time(python3 preprocess.py --input_path ./task_orig/kaggle_set/y_kaggle.csv --output_path ./task/kaggle_set/y_kaggle.$VERSION.parquet --ft_module lvl_transaction.models.features.$VERSION)




# # -- no feature engineering, just copy /clean the files
# time(python3 preprocess.py --input_path ./task_orig/train_set/x_train.csv --output_path ./task/train_set/x_train.ver0.parquet)
# time(python3 preprocess.py --input_path ./task_orig/train_set/y_train.csv --output_path ./task/train_set/y_train.ver0.parquet)

# time(python3 preprocess.py --input_path ./task_orig/val_set/x_val.csv --output_path ./task/val_set/x_val.ver0.parquet)
# time(python3 preprocess.py --input_path ./task_orig/val_set/y_val.csv --output_path ./task/val_set/y_val.ver0.parquet)

# time(python3 preprocess.py --input_path ./task_orig/test_set/x_test.csv --output_path ./task/test_set/x_test.ver0.parquet)

# time(python3 preprocess.py --input_path ./task_orig/kaggle_set/x_kaggle.csv --output_path ./task/kaggle_set/x_kaggle.ver0.parquet)
# time(python3 preprocess.py --input_path ./task_orig/kaggle_set/y_kaggle.csv --output_path ./task/kaggle_set/y_kaggle.ver0.parquet)
