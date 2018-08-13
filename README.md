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
 - [1]: **CNN**: Character-level convolutional networks for text classification ([paper](https://arxiv.org/abs/1509.01626), [code](https://github.com/ArdalanM/nlp-benchmarks/blob/master/src/VDCNN.py))  
 - [2]: **VDCNN**: Very deep convolutional networks for text classification ([paper](https://arxiv.org/abs/1606.01781), [code](https://github.com/ArdalanM/nlp-benchmarks/blob/master/src/CNN.py))  
 - [3]: **HAN**: Hierarchical Attention Networks for Document Classification ([paper](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf), [code]()  

Thanks to [@cedias](https://github.com/cedias) for his **HAN** implentation ([here](https://github.com/cedias/Hierarchical-Sentiment))


## Experiments:
Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper  
 - (ii): Test set accuracy reproduced here  

|                                 | imdb |       ag_news  |     sogu_news     |      db_pedia      | yelp_polarity | yelp_review   | yahoo_answer | amazon_review | amazon_polarity |
|:-------------------------------:|:----:|:--------------:|:-----------------:|:------------------:|:-------------:|:-------------:|:------------:|:-------------:|:---------------:|
|CNN small                        |      | 84.35 / 87.10  | 91.35 / 93.53     | 98.02 / 98.15      |               |               |              |               |                 |
|VDCNN (9 layers, k-max-pooling)  |      | 90.17 / 89.22  | 96.42 / 93.50     | 98.75 / 98.35      | 94.73 / 93.97 | 61.96 / 61.18 |              |               |                 |
|VDCNN (9 layers, max-pooling)    |      | 90.83 / 89.97  | 96.30/ 92.39      | 98.65 /            | 95.12/ 93.47  | 63.27/ 61.21  |              |               |                 |
|VDCNN (17 layers, k-max-pooling) |      | 90.61 / 90.00  | 96.49 /           | - /                | 94.95 / 94.73 | 62.59 /       |              |               |                 |
|VDCNN (17 layers, max-pooling)   |      | 91.12/ 90.09   | 96.46/ 91.93      | 98.60 /            | 95.50/ 94.67  | 63.93/ 62.08  |              |               |                 |
|VDCNN (29 layers, k-max-pooling) |      | 91.33 / 91.22  | 96.82/            | - /                | 95.37 / 94.82 | 63.00 /       |              |               |                 |
|VDCNN (29 layers, max-pooling)   |      | 91.27/ 90.43   | 96.64/ 87.90      | 98.71 /            | 95.72/ 94.75  | 64.26/ 62.73  |              |               |                 |
|    HAN                          | /86.2|                |                   |    / 98.8          |      / 95.0   |               |              |               |                 |

