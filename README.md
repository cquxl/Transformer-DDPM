# Transformer-DDPM

Transformer-DDPM: Anomaly Detection of Time Series with Transformer-based Denoising Diffusion Probablistic Models

* a novel unsupervised Transformer-based Denoising Diffusion Probabilistic Model
* A Gaussian Copula in the forward process of the diffusion model to simulate the joint distribution of noise
* Transformer-DDPM achieves more favorable anomaly detection results against previous well-established methods

# Get Start

* Install Python 3.9, Pytorch == 1.11.0
* Download data, such as SMD, PSM, MSL and SMAP. For the SWAT, you can apply it for it by the following link [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
* set up the environment

```cmd
# git clone the projects
git clone git@github.com:Walker-xionglang/Transformer-DDPM.git
# install the library
pip install -r requirements.txt
```

* train and evaluate

```cmd
python main.py
```

# Transformer-DDPM backbone is Anomaly Transformer

Our Transformer-DDPM's backbone is built on top of [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer) who won the ICLR 2022 Spotlight. One may refer to the link to receive more details.
