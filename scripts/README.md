##### Dependencies Installation

Scripts currently requires Python 3.4 or later to run. Please install `Python` and
`pip` via the package manager of your operating system if it is not included
already. Also you need R language:

```bash
$ apt-get install r-base
```

And you need `psych` package. To install it run following in R shell:

```R
> install.packages('psych', dependencies=TRUE, repos='http://cran.us.r-project.org')
```

To install needed Python dependencies run next in terminal:

```bash
$ pip3 install -r https://github.com/lodurality/35_methods_MICCAI_2017/blob/master/requirements.txt
```

##### Scripts Running

There are three main scripts in `scripts` directory:

* calculate\_icc.py
* classificate\_gender.py
* classificate\_pairwise.py

An argument for these scripts is a string of format:

```bash
$ ./classificate_pairwise.py dataset-normalization-tractography-reconstruction_model-atlas_type
```

For example:

```bash
$ ./classificate_pairwise.py BNU_1-max-deter-csd-con_aparc68+subcort
```

All calculated combinations you can find in
`combinations_for_gender_classification_and_icc.txt` and
`combinations_for_pairwise_classification.txt'. More information about
parameters you can find in the article.

To test calculated results you can use `test_results.py` script. Just run it
and choose directory you want to check:

```bash
$ ./test_results.py

0 /gender_results/
1 /icc_results/BNU_1/
2 /icc_results/HNU_1/
3 /icc_results/IPCAS_1/
4 /pairwise_results/

Please choose what directories to check:
$ 0
/gender_results/


	1/327 are similar
	0 ain't similar

```

Here we calculated results for one combination and script said us the result
the same as we used for the article.
