# LSTM Spam Classifier: OFAT vs. SMAC3 Optimization

This project explores two distinct strategies for hyperparameter optimization of a Long Short-Term Memory (LSTM) neural network trained on the SMS Spam Collection dataset.

The core objective is to compare a naive, manual **One-Factor-At-A-Time (OFAT)** approach against an automated, state-of-the-art **AutoML** approach using the **SMAC3** (Sequential Model-based Algorithm Configuration) library with Multi-Fidelity optimization.

## Dataset

The project uses the **SMS Spam Collection Dataset**.

* **File:** `spam.csv`
* **Content:** A set of SMS messages tagged as either `ham` (legitimate) or `spam`.
* **Preprocessing:** Tokenization, sequence padding, and Label Encoding are handled within the notebook.

## Dependencies & Installation

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Install SMAC3

SMAC3 is the key requirement for the AutoML portion of this project.

**Note for Google Colab / Linux users:**
SMAC depends on `swig` for its Random Forest backend. If you encounter installation errors, run:

```bash
sudo apt-get update
sudo apt-get install swig
```

For other installation options, please refer to the [official documentation](https://automl.github.io/SMAC3/main/1_installation/).

> **WARNING:** `SMAC` is currently only available to **windows users through WSL** and on an experimental as-is installation. For more information, see [this section](https://automl.github.io/SMAC3/main/10_experimental/) of the documentation.

### 3. Install Standard Libraries

All standard dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Hyperparameter Configuration Space

The following hyperparameters are optimized in both approaches:

| Hyperparameter | Type | Range / Values |
| --- | --- | --- |
| **Dropout** | Float | `0.1` - `0.6` |
| **Layers** | Integer | `1`, `2`, `3` |
| **Embedding Dim** | Integer | `20` - `200` |
| **LSTM Units** | Ordinal | `[16, 20, 24, ... 128]` |
| **Learning Rate** | Float (Log) | `0.0001` - `0.01` |

## Results Comparison

* **OFAT Approach:** Provides clear, interpretable graphs for single parameters but often misses the best global configuration because it ignores how parameters (like *Learning Rate* and *Batch Size*) interact with each other.
* **SMAC3 Approach:** Efficiently navigates the high-dimensional space. By using **Hyperband**, it avoids wasting time on bad models, allowing it to test far more configurations in the same amount of compute time.

## References & Citation

This project utilizes **SMAC3**. If you find this comparison useful or use the SMAC library in your own research, please cite the original paper:

> M. Lindauer, K. Eggensperger, M. Feurer, A. Biedenkapp, D. Deng, C. Benjamins, T. Ruhkopf, R. Sass, and F. Hutter. **SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization.** *Journal of Machine Learning Research (JMLR)*, 23(54):1−34, 2022.

**BibTeX:**

```bibtex
@article{lindauer-jmlr22a,
  author  = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and Andr\'{e} Biedenkapp and Difan Deng and Carolin Benjamins and Tim Ruhkopf and Ren\'{e} Sass and Frank Hutter},
  title   = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {54},
  pages   = {1--34},
  url     = {http://jmlr.org/papers/v23/21-0888.html}
}