# SBI_project
This repo contains the code for a master thesis project about comparing four different generative models' ability for approximate ten different benchmarking posterior distributions, within the context of simulation-based inference.

The main notebook, "main_trainig_n_generating", contains classes for the four generative models; two versions of the Conditional Variational Autoencoder, one with an adaptive latent prior and one with a deterministic latent space. The two other classes are for Masked Autoregressive Flows (MAF) and Neural Spline Flows (NSF). These flow-based models utilize the nflows python package.

Furthermore, the main notebook exploits each class in a general script that trains a given class, using pre-generated training datasets. These are generated using the function "generate_dataset.py". Later, samples from the trained inference model are gathered, saved to an .npz-file and plotted.

The report quantifies each models approximation by Maximum Mean Discrepancy (MMD), in which mean ± std are calculated in the notebook "calculate_mmd".
