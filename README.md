# pyPANTERA
## A Python **P**ackage for n**A**tural la**N**guage obfusca**T**ion **E**nforcing p**R**ivacy & **A**nonymization

<center>

![pyPANTERA Logo](https://raw.githubusercontent.com/Kekkodf/pypantera/main/images/pyPANTER.webp)

</center>

## What is pyPANTERA?

pyPANTERA[^1] is a Python package that provides a simple interface to obfuscate natural language text. It is designed to help privacy practitioners implement, reproduce and test State-of-the-Art techniques for natural language obfuscation that implements $\varepsilon$ -Differential Privacy. The repository offers a unified and flexible framework to test NLP and IR tesks like Sentiment Analysis and Document Retrieval. The package is built using numpy, pandas, and scikit-learn libraries, and it is designed to be easy to use and integrate with other Python packages. 

The package offers a combination of natural language processing and mathematical transformations to obfuscate natural language text. It replaces the original string texts with their obfuscated versions, ensuring that the obfuscated text is not directly related to the original text. The obfuscation is performed using word embeddings and word sampling mechanisms, and it is designed to be $\varepsilon$-Differential Privacy compliant.

## Pipeline in pyPANTERA

<center>

![pyPANTERA Pipeline](https://raw.githubusercontent.com/Kekkodf/pypantera/main/images/Pipeline_pyPANTERA.png)

</center>

## Virtual Environment - First way to run pyPANTERA

We provide also a virtual environment to run the package:
1. You can create the virtual environment ***virtualEnvPyPANTERA*** using the ```environment.yml``` file, and running in your terminal:

```bash
conda env create -f environment.yml
```

2. Once the environment is created, you can verify that it is installed by running:

```bash
conda env list
```

3. Finally, you can activate the virtual environment by running:

```bash
conda activate virtualEnvPyPANTERA
```

In the ```requirements.txt``` file, you can find the list of the requirements used by pyPANTERA.

## PyPI - Alternative way to run pyPANTERA

pyPANTERA is designed to be easy to use and accessible for everyone. You can install it using pip:

```bash
pip install pypantera
```

Once installed, you can use it in your Python code by importing it as follows:

```python
import pypantera
```

## What can pyPANTERA do?

pyPANTERA implements current State-of-the-Art mechanisms that use $\varepsilon$-Differential Privacy to obfuscate natural language text. 

The mechanisms implemented in pyPANTERA are divided into two categories:

- **Word Embeddings Perturbation**: This mechanism uses word embeddings to obfuscate the text. It replaces the original word embeddings with a perturbated version of them. Such perturbation is done by adding a statistical noise depending on the mechanism design. The mechanisms implemented are the following:
    - Calibrated Multivariate Perturbations (**CMP**): Addition of spherical noise to the word embeddings. See reference [^2] for more information.
    - Mahalanobis Perturbations (**Mahalanobis**): Addition of elliptical noise to the word embeddings. See reference [^3] for more information.
    - Vickrey family of mechanisms (**Vickrey**): Perturbation is performed using a threshold value to select the nearest perturbed embedding of a term. See reference [^4] for more information.

- **Word Sampling Perturbation**: This mechanism uses word sampling to obfuscate the text. The mechanism computes for each word in the text a list of neighbouring words with the respective scores, then it samples a substitution candidate by basing such sampling on the scores of the neighbouring terms and the privacy budget $\varepsilon$. The mechanisms implemented are the following:
    - Customized Text (**CusText**): Sampling of the substitution candidate from the neighbouring $k$ words of the original word. See reference [^5] for more information.
    - Sanitization Text (**SanText**): Sampling of the substitution candidate from the neighbouring words of the original word. See reference [^6] for more information.
    - Truncated Exponential Mechanism (**TEM**): Sampling of the substitution candidate using the exponential mechanism with the scores of the neighbouring words. See reference [^7] for more information.

## How does pyPANTERA work?

We provide a simple example to show how pyPANTERA works with a concrete example. We suggest using the prepared virtual environment to run the example and the base script `testTASK.py` to run the obfuscation pipeline, i.e, ObfuscationIR and ObfuscationSentiment.

```bash
python testObfuscationIR.py --embPath /absolute/path/to/embeddings --inputPath /absolute/path/to/input/data --outputPath /absolute/path/to/output/data --mechanism MECHANISM --epsilon EPSILON --task TASK --numberOfObfuscations N --PARAMETERS
    
```

The script will run the obfuscation pipeline using the embeddings in the path provided in the `--embPath | -eP` argument, the input data in the path provided in the `--inputPath | -i` argument, and `--outputPath | -o` is used as output path for storing the results. If `--outputPath` is not provided, it creates a folder `./results/task/mechanism/` to save the obfuscated data frames. 

pyPANTERA requires that the input data is a CSV file with a column named `text` that contains the text to obfuscate and an `id` to keep track of the correspondence between original and obfuscated versions. 

The `--task | -tk` argument is used to specify the future task that you want to perform using the new obfuscated texts. The `--epsilon | -e` argument is used to specify the epsilon value for the differential privacy mechanism. The `--mechanism | -m` argument is used to specify the mechanism to use for the obfuscation. The `--numberOfObfuscations | -n` argument is used to specify the number of obfuscations to perform for the same text. Finally, the `--PARAMETERS` are the parameters for the mechanism that you want to use. We provide a specific list of parameters for each mechanism in the following section.

## UML of pyPANTERA

The UML diagram of the pyPANTERA source code is displayed below:

<center>

![pyPANTERA UML diagram](https://raw.githubusercontent.com/Kekkodf/pypantera/main/images/classes.png)

</center>

## Parameters

The script `test.py` has the following parameters, based on the mechanism parameters that you want to use:

- **General Parameters**:
    - `--embPath | -eP`: The path to the word embeddings file (default str: None, **required**)
    - `--inputPath | -i`: The path to the input data file (default str: None, **required**)
    - `--outputPath | -o`: The path to the output data file (default str: None)
    - `--task | -tk`: The future task that you want to perform using the new obfuscated texts (default str: 'retrieval')
    - `--epsilon | -e`: The epsilon value for the differential privacy mechanism (default List[float]: [1.0, 5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 50.0])
    - `--mechanism | -m`: The mechanism to use for the obfuscation (default str: 'CMP', choices: ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'CusText', 'SanText', 'TEM'])
    - `--numberOfObfuscations | -n`: The number of obfuscations to perform for the same text (default int: 1)

- **CMP**: The parameters for the CMP mechanism are only the general ones.
- **Mahalanobis**: The parameters for the Mahalanobis mechanism are the following:
    - `--lam`: The lambda value for the Mahalanobis norm (default float: 1)
- **VickreyCMP/VickreyMhl**: The parameters for the Vickrey mechanism are the following:
    - `--t`: The threshold value for the Vickrey mechanism (default float: 0.75). Eventually, if you use the `VickreyMhl` mechanism, you can also use the `--lam` parameter to set the lambda value for the Mahalanobis norm (default float: 1)
- **CusText**: The parameters for the CusText mechanism are the following:
    - `--k`: The number of neighbouring words to consider for the sampling (default int: 10)
    - `--distance | -d`: The distance metric to use for the sampling (default str: 'Euclidean')
- **SanText**: The parameters for the SanText mechanism are only the general ones.
- **TEM**: The parameters for the TEM mechanism are the following:
    - `--beta`: The beta value for the exponential mechanism (default float: 0.001)

## Example

Suppose you want to run the obfuscation pipeline using the `CMP` mechanism with the embeddings in the path `./embeddings/glove.6B.50d.txt`, the input data in the path `./data/input.csv`, and the output data in the path `./data/output.csv`, for all the default values of $\varepsilon$ obtaining only one obfuscation for the original text. You can run the following command:

```bash
python test.py --embPath /embeddings/glove.6B.50d.txt --inputPath /data/input.csv --outputPath /data/output/ --mechanism CMP
```

To enhance the clarity of how pyPANTERA works, we add a toy Python Notebook to simulate some of the obfuscation implemented. 

## Experimental configuration of the mechanisms
The following Table reports the experimental parameters used for obtaining the obfuscated texts used in the experiments presented in the paper.


<table>
<tr><th>Embedding Perturbation Mechanisms </th><th>Sampling Perturbation Mechanisms</th></tr>
<tr><td>

| **Mechanism**  | **Parameters**        |
|----------------|-----------------------|
| CMP            | -                     |
| Mahalanobis    | $\lambda=1$           |
| VickreyCMP     | $t=0.75$              |
| VickreyMhl     | $t=0.75$, $\lambda=1$ |
</td><td>

| **Mechanism**  | **Parameters**        |
|----------------|-----------------------|
| CusText        | $K=10$                |
| SanText        | -                     |
| TEM            | $\beta=0.001$         |

</td></tr> </table>

## Final results overview

Using the `test.py` script running CMP, embeddings 300d GloVe with the default parameters, we obtain the following results for the DL'19 queries dataset (overview of the first two rows, for $\varepsilon = 1, 5, 10$):

| id | text | obfuscatedText | mechansim | epsilon | 
|----|------|-----------------|-----------|---------|
| 156493 | do goldfish grow | hipc householder 1976-1983|CMP|1|
|1110199|what is wifi vs bluetooth|25-june nonsubscribers trimet edema ---|CMP|1|

| id | text | obfuscatedText | mechansim | epsilon | 
|----|------|-----------------|-----------|---------|
| 156493 | do goldfish grow | foil householder scotland|CMP|5|
|1110199|what is wifi vs bluetooth| galangal naat trimet edema ---|CMP|5|

| id | text | obfuscatedText | mechansim | epsilon | 
|----|------|-----------------|-----------|---------|
| 156493 | do goldfish grow | do goldfish grow | CMP|10|
|1110199|what is wifi vs bluetooth|out salvage terrestrial 7-3 bluetooth|CMP|10|

## License

The package is released under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. You can find the full text of the license in the LICENSE file.



[^1]: Prompt for DALL-E pyPANTER generation: "A cute panther sitting beside the Python programming language symbol. The panther should have big, expressive eyes and a friendly demeanor, sitting in."

[^2]: [Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations](https://dl.acm.org/doi/10.1145/3336191.3371856) (Feyisetan et al., In Proceedings of the International Conference on Web Search and Data Mining, 2020)

[^3]: [A Differentially Private Text Perturbation Method Using Regularized Mahalanobis Metric](https://aclanthology.org/2020.privatenlp-1.2) (Xu et al., In Proceedings of the Second Workshop on Privacy in NLP, 2020)

[^4]: [On a Utilitarian Approach to Privacy Preserving Text Generation](https://aclanthology.org/2021.privatenlp-1.2) (Xu et al., In Proceedings of the Third Workshop on Privacy in Natural Language Processing, 2021)

[^5]: [A Customized Text Sanitization Mechanism with Differential Privacy](https://aclanthology.org/2023.findings-acl.355) (Chen et al., In Findings of the Association for Computational Linguistics, 2023)

[^6]: [Differential Privacy for Text Analytics via Natural Text Sanitization](https://aclanthology.org/2021.findings-acl.337) (Yue et al., In Findings of the Association for Computational Linguistics, 2021)

[^7]: [TEM: High Utility Metric Differential Privacy on Text]() (Carvalho et al., In Proceedings of the 2023 SIAM International Conference on Data Mining, 2023)
