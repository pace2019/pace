# Practical Accuracy Estimation for Efficient DNN Testing
This is the homepage of **PACE** including `tool implementation`, `evaluation scripts`, `studied DNN models` , `corresponding testing sets` and `experiment results`. 

#### Environment configuration
Before running PACE, please make sure you have installed various related packages, including keras, tensorflow, hdbscan and sklearn.

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

#### Datasets and pre-trained models

We published all studied DNN models we utilized and you can find them in `mnist_cifar_imagenet_svhn\model`.

Meanwhile, we released all corresponding testing sets in the `mnist_cifar_imagenet_svhn\data`. The data of MNIST, CIFAR-10 and CIFAR-100 can be obtained directly from Keras API.

Regarding to Driving, the pre-trained models can be found in folder `driving`, and the testing sets are in the `driving\testing`.

