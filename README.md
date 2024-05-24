# Transfer-learning

Transfer learning and fine-tuning implementations on CIFAR-10 datasets with TensorFlow and PyTorch.

-----
### Train, validation & test datasets:
The training dataset is split into a training set and a validation set. The training set is used to train and validate the model during the training step, while the test dataset is used to evaluate the model after the training step.

In TensorFlow, it is as easy as the following code:

```Python
train_datasets, test_datasets = tf.keras.datasets.cifar10.load_data()

# split the train datasets into train & validation datasets
val_size       = 10000
val_datasets   = (train_datasets[0][:val_size], train_datasets[1][:val_size])
train_datasets = (train_datasets[0][val_size:], train_datasets[1][val_size:])
 ```

However, it is a bit trickier in PyTorch (e.g. here: https://stackoverflow.com/questions/61811946/train-valid-test-split-for-custom-dataset-using-pytorch-and-torchvision), especially if different transforms should be applied to validation datasets.

In the following code, a more straightforward approach is considered:
```Python
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=train_transforms)
valset   = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=test_val_transforms)
testset  = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=test_val_transforms)

# split the train datasets into train & validation datasets
val_size      = 10000
valset.data   = trainset.data[:val_size]; valset.targets   = trainset.targets[:val_size]
trainset.data = trainset.data[val_size:]; trainset.targets = trainset.targets[val_size:]
 ```
