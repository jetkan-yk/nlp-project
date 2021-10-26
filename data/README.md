# Dataset

The dataset is organized as follows:

    [task-name-folder]/                                # natural_language_inference, paraphrase_generation, question_answering, relation_extraction, topic_models
        ├── [article-counter-folder]/                  # ranges between 0 to 100 since we annotated varying numbers of articles per task
        │   ├── [article-name].pdf                      # scholarly article pdf
        │   ├── [article-name]-Grobid-out.txt           # plaintext output from the [Grobid parser](https://github.com/kermitt2/grobid)
        │   ├── [article-name]-Stanza-out.txt           # plaintext preprocessed output from [Stanza](https://github.com/stanfordnlp/stanza)
        │   ├── sentences.txt                          # annotated Contribution sentences in the file
        │   ├── entities.txt                           # annotated entities in the Contribution sentences
        │   └── info-units/                            # the folder containing information units in JSON format
        │   │   └── research-problem.json              # `research problem` mandatory information unit in json format
        │   │   └── model.json                         # `model` information unit in json format; in some articles it is called `approach`
        │   │   └── ...                                # there are 12 information units in all and each article may be annotated by 3 or 6
        │   └── triples/                               # the folder containing information unit triples one per line
        │   │   └── research-problem.txt               # `research problem` triples (one research problem statement per line)
        │   │   └── model.txt                          # `model` triples (one statement per line)
        │   │   └── ...                                # there are 12 information units in all and each article may be annotated by 3 or 6
        │   └── ...                                    # there are K articles annotated for each task, so this repeats for the remaining K-1 annotated articles
        └── ...                                        # there are 38 task folders overall, then this repeats 37 more times
