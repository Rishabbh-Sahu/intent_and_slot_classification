# <p align="center"> Intent and slot classification </p>
#### About this project

Large transformers Bert, Roberta, XLnet .. a huge list, isn't it? Question we should ask ourselves: do we seriously need over 100's of Mn parameters to do classification or other similar tasks? Do we need these many attention layers?

Answer lies within the amount of data to finetune i.e. data points to distinguish the number of unique classes and numerous patterns we may want to support. However, large transformers in general could take days to train and need massive amounts of data hence a long delay to access model performance. Getting this label data is not only very time consuming however resource intensive too.

A very straightforward solution is to use small/tiny models, **as small as having 2 attention layers with size <20MB**. Now training few layers reduces training time significantly and we can gauge the model performance in a few hours. As a result, **the scalability and deployment of NLP-based systems across the industry will become more attainable**.

In order to demonstrate the same, I've choosed one of the very common NLU task is to understand intent (sequence classification) and slots (entities within the sequence) of a dialog and this repo helps you achieve the same with **very little resource settings**. By extracting this knowledge about the queries/patterns, you can capture the context and take appropriate decisions. This repo help classify both together using **Joint model architecture (multitask model)**. Pretrained model **BERT_SMALL** is used to achieve transfer learning however it can be changed to any other BERT variants.

To enable **GPU** support, please do enable **CUDA-11** in windows/linux/mac virtual environment for tf2.4 or use CUDA 10.1 for tf2.3. 

A quike guide for installing TensorFlow version 2.4 with CUDA 11, cudNN 8 and GPU support: step by step tutorial 2021 - https://www.youtube.com/watch?v=hHWkvEcDBO0

#### Getting started
- create virtual environment
- install tensorflow==**2.4**
- install requirements 
- Open config.yaml file and modify parameters as per your setup

#### For training
```
python training.py 
```
With **HF Fast-tokenizer** - To expedite training data pre-processing by leveraging offset_mapping to create tags as per tokenizer's sub-token split, use below command
```
python training_fastTokenizer.py 
```
#### Scores
After 5-epoch
- **Training accuracy : sequence acc ~ 99% and slot acc ~96%**
- **Validation accuracy : sequence acc ~ 99% and slot acc ~96%**<br>
The above results shows, we might not need large transformer model (certainly 100Mn parameters are way too much) yet achieving astounding results. 

#### Inference
For quering the model, use the below command to open **flaskAPI**- <br>
```
python flask_api.py
```
The above command opens up the local server which is accessible thru Postman app or CURL command (listed below). Port can be changed in the script and same can be used while quering the model. <br>
- Input format to query the model via **Postman App**- <br>
![image](https://user-images.githubusercontent.com/69572197/130914705-064dcd6e-99b2-4c3e-a386-92915fc92187.png)<br>
- Input format to query the model via **CURL Command**- <br>
curl --location --request POST 'http://ip-address-of-your-mc:8888/predict' --header 'Content-Type: application/json' --data-raw '{"utterance": "what is the weather in london"}'<br>

*Change utterance value to test different queries*

#### Future scope
1) Publish training accuracies using different benchmark data set called ATIS (Data pertaining to Flight domain)
2) Support for various models like albert, mobilebert, distilbert etc.  

#### Acknowledgement
@article https://arxiv.org/pdf/1805.10190.pdf
{coucke2018snips, title = {Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces}, author = {Coucke, Alice and Saade, Alaa and Ball, Adrien and Bluche, Th{'e}odore and Caulier, Alexandre and Leroy, David and Doumouro, Cl{'e}ment and Gisselbrecht, Thibault and Caltagirone, Francesco and Lavril, Thibaut and others}, journal = {arXiv preprint arXiv:1805.10190}, pages = {12--16}, year = {2018}}

