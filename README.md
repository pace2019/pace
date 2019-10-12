# Practical Accuracy Estimation for Efficient DNN Testing
This is the code implementation of PACE.

Before running, please make sure you have installed various related packages, including keras, tensorflow, hdbscan and sklearn.

You can install hdbscan with the following command：

```shell
pip install hdbscan
```

Please use the following command to execute PACE:

```shell
python -u -m mnist_cifar_imagenet_svhn.selection --exp_id=lenet1 --select_layer_idx=-3 --dec_dim=8 --min_samples=4  --min_cluster_size=80
```

\* Exp_id is the id of the model. Select_layer_idx is the number of layers of the extracted feature. Dec_dim is the number of dimensionality reductions. Min_samples and min_cluster_size are the parameters required for hdbscan clustering.

Also, we put the raw data results for all experiments in the  AllResult folder.