# Adam: A Method for Stochastic Optimization — Reproduction Report

## 1. Paper Summary
The paper "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2015) introduces a novel first-order gradient-based optimization algorithm. The core claim of the paper is that Adam combines the advantages of AdaGrad (which handles sparse gradients well) and RMSProp (which handles non-stationary objectives well). Adam is computationally efficient, has small memory footprint, is invariant to diagonal rescaling of gradients, and naturally performs step size annealing. The novelty lies in the use of adaptive estimates of both lower-order moments (first and second) of the gradients, and critically, the inclusion of bias-correction terms to counteract the initialization of these moving averages at zero.

## 2. Method
Adam maintains exponential moving averages of the gradient ($m_t$) and the squared gradient ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Where $g_t$ is the gradient at timestep $t$, and $\beta_1, \beta_2$ are the decay rates. Since $m_0$ and $v_0$ are initialized to zeros, these moment estimates are biased towards zero, especially during early timesteps. 

The expected value of the second moment estimate relates to the true second moment $\mathbb{E}[g_t^2]$ by:
$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t)$$

To correct this initialization bias, Adam computes the bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The final parameter update is:
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where $\alpha$ is the step size. This approach provides an effective step size that is approximately bounded by $\alpha$ and establishes a trust region around the current parameter value. (Reference: Optimizers, Chapter 27).

## 3. Experimental Setup
Our reproduction leverages the **PyTorch** deep learning framework, enabling GPU acceleration (if available) and highly optimized tensor operations.
- **Hardware**: CPU / NVIDIA GPU.
- **Datasets**: MNIST and CIFAR-10 (`torchvision.datasets`).
- **Models**:
  - *Logistic Regression (MNIST)*: `nn.Linear(784, 10)`.
  - *MLP (MNIST)*: Two hidden layers of 1000 units with `nn.ReLU` and 50% `nn.Dropout`.
  - *CNN (CIFAR-10)*: Three stages of 5x5 `nn.Conv2d` and 3x3 `nn.MaxPool2d` with stride 2. `c64-pool-c64-pool-c128-pool` followed by a fully connected layer of 1000 units and 50% Dropout.
  - *VAE (MNIST)*: Encoder/decoder architecture with a single 500-unit hidden layer and Softplus nonlinearities, mapping to a 50-dimensional latent space.
- **Optimizers**: We used `torch.optim.Adam`, `torch.optim.SGD`, `torch.optim.Adagrad`, and `torch.optim.RMSprop`. For the bias-correction ablation study, we implemented a custom `AdamUncorrected` subclass. We used the paper's recommended defaults: Adam ($\alpha=0.001, \beta_1=0.9, \beta_2=0.999$).

## 4. Results
> [!NOTE]
> View the generated plots in the `results/` folder.
* **Logistic Regression**: Adam and SGD+Nesterov exhibit very similar convergence profiles, significantly outperforming AdaGrad, which mirrors Figure 1 of the paper.
* **MLP with Dropout**: Adam outperforms RMSProp, AdaGrad, and SGD, achieving lower training cost faster, reflecting the findings in Figure 2a.
* **CNN**: Adam adapts its learning rate for the varied gradients across convolutional layers, achieving rapid convergence. The results closely follow Figure 3.

Our results align with the original paper's reported trends within the expected tolerance, demonstrating Adam's robust optimization across varied architectures.

## 5. Ablation: Bias-Correction
We reproduced the paper's bias-correction ablation study (Figure 4) by training a VAE on MNIST. We varied $\beta_2 \in \{0.99, 0.999, 0.9999\}$ and $\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$.
The uncorrected Adam exhibited extreme instability and divergence during the first 10 epochs when $\beta_2$ approached 1.
This is because a large $\beta_2$ (e.g. 0.9999) causes the initial uncorrected second moment estimate $v_t$ to remain extremely close to 0. Consequently, dividing by $\sqrt{v_t}$ results in massive, destabilizing parameter updates. The bias correction $\frac{v_t}{1 - \beta_2^t}$ correctly scales the estimate up, neutralizing this initial instability and ensuring smooth convergence.

## 6. Reproducibility
- Because we utilized PyTorch, the experimental suite runs in a fraction of the time compared to raw Python/NumPy, enabling us to run all models on their full datasets (60,000 images) for all specified epochs without subsampling.
- Run `python main.py --all` to execute the entire deterministic pipeline and generate all standard figures.
