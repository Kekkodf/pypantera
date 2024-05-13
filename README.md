# pyPANTERA
## Python **P**ackage for n**A**tural la**N**guage obfusca**T**ion **E**nforcing p**R**ivacy & **A**nonymization
<p align="center">
    <img src="./images/pyPANTER.webp" width="255">
</p>

## What is pyPANTERA?

pyPANTERA[^1] is a Python package that provides a simple interface to obfuscate natural language text. It is designed to help developers and data scientists to implement, reproduce and test state-of-the-art techniques for natural language obfuscation that implements $\varepsilon$ -Differential Privacy. The package is built using numpy, pandas, and scikit-learn libraries, and it is designed to be easy to use and integrate with other Python packages. 

The package offers a combination of natural language processing and mathematical transformations to obfuscate natural language text. It replaces the opriginal string texts 

## How to use pyPANTERA?

pyPANTERA is designed to be easy to use and accessible for everyone. You can install it using pip:

```bash
pip install pypantera
```

Once installed, you can use it in your Python code by importing it as follows:

```python
import pypantera
```

## Virtual Environment

We provide also a virtual environment to run the package. You can create the virtual environment ***virtualEnvPyPANTERA*** using the ```environment.yml``` file, and running in your terminal:

```bash
conda env create -f environment.yml
```

Once the environment is created, you can verify that it is installed by running:

```bash
conda env list
```

Finnally, you can activate the virtual environment by running:

```bash
conda activate virtualEnvPyPANTERA
```

In the ```requirements.txt``` file you can find the list of the packages exported from the virtual environment.

## What can pyPANTERA do?

pyPANTERA implements current state of the art mechanisms that uses $\varepsilon$-Differential Privacy to obfuscate natural language text. 

The mechansims implemented in pyPANTERA are divided in two categories:

- **Word Embeddings Perturbation**: This mechanism uses word embeddings to obfuscate the text. It replaces the original words ebeddings with a perturbated version of them. Such perturbation is done by adding a statistical noise depending on the mechansim design. The mechansim implemented are the following:
    - Calibrated Multivariate Perturbations (**CMP**): Addition of sferical noise to the word embeddings. See reference [^2] for more information.
    - Mahalanobis Perturbations (**Mahalanobis**): Addition of eliptical noise to the word embeddings. See reference [^3] for more information.
    - Vickrey family of mechanisms (**Vickrey**): Perturbation performed using a treshold value to select the nearest perturbed embedding of a term. See reference [^4] for more information.

## Why use pyPANTERA?

pyPANTERA is designed to help Data Scientists and Researchers to protect the privacy of their data by obfuscating sensitive texts. It can be used to obfuscate sensitive natural language texts by implemneting current state of the art techniques of text obfuscation based o Differential Privacy and Word Embeddings.


## License

The package is released under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. You can find the full text of the license in the LICENSE file.



[^1]: Prompt for DALL-E pyPANTER generation: "A cute panther sitting beside the Python programming language symbol. The panther should have big, expressive eyes and a friendly demeanor, sitting in."

[^2]: [Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations](https://dl.acm.org/doi/10.1145/3336191.3371856) (Feyisetan et al., In Proceedings of the International Conference on Web Search and Data Mining, 2020)

[^3]: [A Differentially Private Text Perturbation Method Using Regularized Mahalanobis Metric](https://aclanthology.org/2020.privatenlp-1.2) (Xu et al., In Proceedings of the Second Workshop on Privacy in NLP, 2020)

[^4]: [On a Utilitarian Approach to Privacy Preserving Text Generation](https://aclanthology.org/2021.privatenlp-1.2) (Xu et al., In Proceedings of the Third Workshop on Privacy in Natural Language Processing, 2021)

