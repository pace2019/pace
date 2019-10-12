# Practical Accuracy Estimation for Efficient DNN Testing
This is the code implementation of PACE.

#### Environment configuration
Before running, please make sure you have installed various related packages, including keras, tensorflow, hdbscan and sklearn.

You can install hdbscan with the following command：

```shell
pip install hdbscan
```

#### Running
Please use the following command to execute PACE:

```shell
python -u -m mnist_cifar_imagenet_svhn.selection --exp_id=lenet1 --select_layer_idx=-3 --dec_dim=8 --min_samples=4  --min_cluster_size=80
```

- `exp_id` : the id of the model
- `select_layer_idx` : index of layer which is selected to extract feature 
- `dec_dim` : the dimension after reduction 
- `min_samples` and `min_cluster_size` : the parameters required for hdbscan clustering

#### Results
Also, we put the raw data results for all experiments in `AllResult`. 