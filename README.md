## Acoustic Word Embeddings and Phonological Similarity :tea:

This is the code base for the acoustic word embedding models, training experiments, and evaluation scripts for the experiments reported in our **INTERSPEECH 2021** paper 

:pencil: [Do Acoustic Word Embeddings Capture Phonological Similarity? An Empirical Study](https://arxiv.org/pdf/2106.08686.pdf)

<!-- To cite the paper

```
@inproceedings{Abdullah2021DoAW,
  title={Do Acoustic Word Embeddings Capture Phonological Similarity? An Empirical Study},
  author={Badr M. Abdullah and Marius Mosbach and Iuliia Zaitova and Bernd Möbius and Dietrich Klakow},
  booktitle={Proc. Interspeech},
  year={2021}
}
``` -->

### Dependencies :dna:

python 3.8, pytorch 1.1, numpy, scipy, faiss, pickle, pandas, yaml


### Speech Data :speech_balloon: :left_speech_bubble:
The data in our study is drawn from the Multilingual GlobalPhone speech database for  German :de: and  Czech :czech_republic:. Because the data is distributed under a research license by Appen Butler Hill Pty Ltd., we cannot re-distribute the raw speech data. However, if you have already access to the GlobalPhone speech database and you would have access to our word-alignment annotations, train/test splits, and word-level IPA transcriptions, please contact the first author. 


### Working with the code :snake:
To run a training experiment, write down all hyperparameters and other info in the config file ```config_file_train_awe_bigru_seq2seq.yml```

Then ...

```
>>> cd AWEs_phon_sim
>>> python nn_train_seq2seq_embeddings.py config_files/config_file_train_awe_bigru_seq2seq.yml
```

To evaluate the model on the acoustic word discrimination task, make sure the path to the pre-trained model is in this config file ```config_file_eval_awe_bigru_seq2seq.yml```

Then ...


```
>>> cd AWEs_phon_sim
>>> python nn_eval/nn_eval_seq2seq_embeddings.py config_files/config_file_eval_awe_bigru_seq2seq.yml
```

The code is fairly documented and the vectorization logic, as well as the code for the models, should be useful for other speech technology tasks. If you use our code and encounter problems, please create an issue or contact the first author. 


If you use our code in a work that leads to a publication, please cite our paper as 

```
@inproceedings{Abdullah2021DoAW,
  title={Do Acoustic Word Embeddings Capture Phonological Similarity? An Empirical Study},
  author={Badr M. Abdullah and Marius Mosbach and Iuliia Zaitova and Bernd Möbius and Dietrich Klakow},
  booktitle={Proc. Interspeech},
  year={2021}
}
```