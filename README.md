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
 - [1]: **CNN**: Character-level convolutional networks for text classification ([paper](https://arxiv.org/abs/1509.01626)) 
 - [2]: **VDCNN**: Very deep convolutional networks for text classification ([paper](https://arxiv.org/abs/1606.01781))
 - [3]: **HAN**: Hierarchical Attention Networks for Document Classification ([paper](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)), all credits goes to [@cedias](https://github.com/cedias/Hierarchical-Sentiment)
 - [4]: **Transformer Encoder**: Attention Is All You Need (encoder part) ([paper](https://arxiv.org/abs/1706.03762)), credits to Yu-Hsiang Huang's [work](https://github.com/jadore801120/attention-is-all-you-need-pytorch))

HAN word (red) and sentence (blue) attention weight at prediction:

![](https://media.giphy.com/media/1QgFc8oW7m600Cvv5D/giphy.gif)

## Experiments:
Results are reported as follows:  (i) / (ii)
 - (i): Test set accuracy reported by the paper  
 - (ii): Test set accuracy reproduced here  

### Imdb
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |                |                |
| VDCNN 9 layers  |                |                |
| VDCNN 17 layers |                |                |
| VDCNN 29 layers |                |                |
| HAN             |                |    **90.5**    |
| Transformer     |                |     84.1       |


### Ag news 
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |    84.35       |88.30           |
| VDCNN 9 layers  |    90.17       |  89.22         |
| VDCNN 17 layers |  90.61         |  90.00         |
| VDCNN 29 layers |  **91.27**     |     90.43      |
| HAN             |                |      **92.4**  |
| Transformer     |                |     **92.4**       |



### Sogu news
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |   91.35        |   93.53        |
| VDCNN 9 layers  |    96.42       |   93.50        |
| VDCNN 17 layers |     96.49      |                |
| VDCNN 29 layers |    **96.64**     | 87.90          |
| HAN             |                |   **96.**       |
| Transformer     |                |   95.6       |


### DBpedia
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |98.02           | 98.15          |
| VDCNN 9 layers  | **98.75**      | 98.35          |
| VDCNN 17 layers |98.02           | 98.15          |
| VDCNN 29 layers |98.71           |                |
| HAN             |                |   **99.0**     |
|                 |                |                |


### Yelp polarity
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |                |                |
| VDCNN 9 layers  |94.73           | 93.97          |
| VDCNN 17 layers |94.95           | 94.73          |
| VDCNN 29 layers |**95.72**       |  **94.75**     |
| HAN             |                |                |
|                 |                |                |


### Yelp review
| Model           | paper accuracy | repo accuracy  |
|:---------------:| :-------------:| :------------- |
| CNN small       |                |                |
| VDCNN 9 layers  |61.96           | 61.18          |
| VDCNN 17 layers |62.59           |                |
| VDCNN 29 layers |**64.26**       |  62.73         |
| HAN             |                |  **63.**       |
|                 |                |                |
