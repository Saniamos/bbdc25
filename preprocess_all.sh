VERSION="ver01"

time(python3 preprocess.py --input_path ./task_orig/train_set/x_train.csv --output_path ./task/train_set/x_train.$VERSION.csv --ft_module models.features.$VERSION)
time(python3 preprocess.py --input_path ./task_orig/train_set/y_train.csv --output_path ./task/train_set/y_train.$VERSION.csv --ft_module models.features.$VERSION)

time(python3 preprocess.py --input_path ./task_orig/val_set/x_val.csv --output_path ./task/val_set/x_val.$VERSION.csv --ft_module models.features.$VERSION)
time(python3 preprocess.py --input_path ./task_orig/val_set/y_val.csv --output_path ./task/val_set/y_val.$VERSION.csv --ft_module models.features.$VERSION)

time(python3 preprocess.py --input_path ./task_orig/kaggle_set/x_kaggle.csv --output_path ./task/kaggle_set/x_kaggle.$VERSION.csv --ft_module models.features.$VERSION)
time(python3 preprocess.py --input_path ./task_orig/kaggle_set/y_kaggle.csv --output_path ./task/kaggle_set/y_kaggle.$VERSION.csv --ft_module models.features.$VERSION)

time(python3 preprocess.py --input_path ./task_orig/test_set/x_test.csv --output_path ./task/test_set/x_test.$VERSION.csv --ft_module models.features.$VERSION)