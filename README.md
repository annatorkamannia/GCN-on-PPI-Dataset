# GCN-on-PPI-Dataset
Implementation of the Graph Convolutional Network (GCN) for the Protein-Protein Interaction (PPI) dataset

<p>This repository contains the implementation of a Graph Convolutional Network (GCN) for the Protein-Protein Interaction (PPI) dataset using PyTorch Geometric. The goal of this project is to predict the labels of proteins by utilizing the graph structure of their interactions.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#description">Description</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model">Model</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
</ul>

<h2 id="description">Description</h2>
<p>Graph Convolutional Networks (GCNs) have shown great promise in modeling graph-structured data, particularly for tasks like node classification. In this project, we apply a GCN to the PPI dataset, where the goal is to predict protein functions. This project is a hands-on practice for implementing GCNs on a real-world biological dataset.</p>

<h2 id="installation">Installation</h2>
<p>To run this project, you need to have the following installed:</p>
<ul>
    <li>Python 3.8+</li>
    <li>PyTorch</li>
    <li>PyTorch Geometric</li>
</ul>
<ol>
    <li>Clone this repository:</li>
    <pre><code>git clone https://github.com/annatorkamannia/GCN-on-PPI-Dataset </br>cd gcn-ppi</code></pre>
    <li>Install dependencies:</li>
    <pre><code>pip install -r requirements.txt</code></pre>
    <li>Install PyTorch Geometric following the official installation instructions for your system and CUDA version:</li>
    <a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html">PyTorch Geometric Installation Guide</a>
</ol>

<h2 id="usage">Usage</h2>
<p>To train the GCN model on the PPI dataset:</p>
<ol>
    <li>Download the dataset (PPI will automatically be downloaded by the script).</li>
    <li>Run the training script:</li>
    <pre><code>python train.py</code></pre>
    <p>The model will train for 100 epochs, and the validation accuracy will be displayed for each epoch.</p>
</ol>
<p>To evaluate the model on the test set:</p>
<pre><code>python test.py</code></pre>

<p>To save the trained model:</p>
<pre><code>torch.save(model.state_dict(), 'gcn_model_ppi.pth')</code></pre>

<h2 id="model">Model</h2>
<p>The GCN model is composed of 4 graph convolutional layers. Each layer learns node representations by aggregating information from the neighboring nodes. The model is trained using binary cross-entropy loss to handle multi-label classification, and the output layer uses a sigmoid activation to predict multiple labels for each node.</p>
<ul>
    <li>Layer 1: Input: 50 features (from PPI dataset) → 128 hidden units</li>
    <li>Layer 2: 128 hidden units → 128 hidden units</li>
    <li>Layer 3: 128 hidden units → 128 hidden units</li>
    <li>Layer 4: Output: 121 labels (multi-label classification)</li>
</ul>

<h2 id="results">Results</h2>
<p>After training for 100 epochs, the model achieves a reasonable validation accuracy on the PPI dataset. The final test accuracy is printed out at the end of training. The accuracy measures how well the model predicts the multi-label classifications for each protein in the graph.</p>

<h3>Sample Output</h3>
<pre><code>
Epoch 100, Loss: 0.4272, Validation Accuracy: 26.15%
Test Accuracy: 22.86%
</code></pre>

<h2 id="dataset">Dataset</h2>
<p>The PPI dataset consists of multiple graphs representing protein-protein interactions. The task is to predict protein functions (multi-label classification). The dataset contains train, validation, and test splits. It can be downloaded automatically using the <code>torch_geometric.datasets.PPI</code> module.</p>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! If you have suggestions for improvements, feel free to submit a pull request or open an issue.</p>

