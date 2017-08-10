# nlp-benchmark

## Datasets:
| Dataset                | Classes | Train samples | Test samples | source |
|------------------------|:---------:|:---------------:|:--------------:|:--------:|
| Imdb                   |    2    |    25 000     |     25 000   |[link](https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/imdb_csv.tar.gz)|
| AG’s News              |    4    |    120 000    |     7 600    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Sogou News             |    5    |    450 000    |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| DBPedia                |    14   |    560 000    |    70 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yelp Review Full       |    5    |    650 000    |    50 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |[link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|


## Models:
 - [1]: **CNN**: Character-level convolutional networks for text classification ([paper](https://arxiv.org/abs/1509.01626), [code]())  
 - [2]: **VDCNN**: Very deep convolutional networks for text classification ([paper](https://arxiv.org/abs/1606.01781), [code]())  


## Experiments:
Results are reported as follows:  (i) / (ii) / (iii)
 - (i): Test set accuracy claimed by the paper  
 - (ii): Test set accuracy reproduced here  
 - (iii): Epoch of convergence reproduced  

|                     | imdb |       ag_news      |     sogu_news     |      db_pedia      | yelp_polarity | yelp_review | yahoo_answer | amazon_review | amazon_polarity |
|---------------------|------|:------------------:|:-----------------:|:------------------:|:-------------:|:-----------:|:------------:|:-------------:|:---------------:|
|CNN small            |      | 84.35 / 87.10 / 45 |91.35 / 93.53 / 05 | 98.02 / 98.15 / 25 |               |             |              |               |                 |
|VDCNN (9 layers)     |      | 90.83 / 89.14 / 23 |96.30 / 93.50 / 50 | 98.75 / 98.35 / 24 |               |             |              |               |                 |
|VDCNN (17 layers)    |      | -/-/-              |      -/           | - /                |               |             |              |               |                 |
|    HAN              |      |                    |                   |                    |               |             |              |               |                 |