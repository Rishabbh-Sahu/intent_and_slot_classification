# intent_and_slot_classification
#### About this project
One of the main NLU tasks is to understand the intent (sequence classification) and slots (entities within the sequence) and this repo helps you do the same. By extracting this knowledge about the queries, you can comprehend the context and take appropriate decisions. This repo help classify both together using Joint Model architecture (multitask model). Pretrained model BERT_SMALL is used to achieve transfer learning however it can be changed to any other BERT variants. Supports for various models like albert, mobilebert etc. would be availalble soon!. 

To enable GPU support, please do enable CUDA-11 in windows/linux/mac virtual environment for tf2.4 or use CUDA 10.1 for tf2.3.

#### Getting started
- create virtual environment
- install tensorflow==2.4
- install requirements 
- Open config.yaml file and modify parameters as per your setup

#### For training
- python training.py 

#### Score
- Training accuracy after 5-epoch : sequence acc ~ 99% and slot acc ~96%
- Validation accuracy : sequence acc ~ 99% and slot acc ~96%

#### Acknowledgement:
@article https://arxiv.org/pdf/1805.10190.pdf
{coucke2018snips, title = {Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces}, author = {Coucke, Alice and Saade, Alaa and Ball, Adrien and Bluche, Th{'e}odore and Caulier, Alexandre and Leroy, David and Doumouro, Cl{'e}ment and Gisselbrecht, Thibault and Caltagirone, Francesco and Lavril, Thibaut and others}, journal = {arXiv preprint arXiv:1805.10190}, pages = {12--16}, year = {2018}}

