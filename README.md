# CS4248 G17 NLP Project

## Project Structure

### Dataset

There are 2 dataset folders:

1. `data/` contains 38 task folders
2. `data-small/` a subset of `data/`, contains 5 task folders

The dataset is organized as follows:

    [task-name-folder]/                                # natural_language_inference, paraphrase_generation, question_answering, relation_extraction, topic_models
        ├── [article-counter-folder]/                  # ranges between 0 to 100 since we annotated varying numbers of articles per task
        │   ├── [article-name].pdf                     # scholarly article pdf
        │   ├── [article-name]-Grobid-out.txt          # plaintext output from the [Grobid parser](https://github.com/kermitt2/grobid)
        │   ├── [article-name]-Stanza-out.txt          # plaintext preprocessed output from [Stanza](https://github.com/stanfordnlp/stanza)
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
        └── ...                                        # if there are N task folders overall, then this repeats N-1 more times

## Project Requirement

The breakdown of the final project marks will be according to the following criteria:

- **Content (50%)**: Is the work valid and technically sound? Is the work significant and of impact? Is relevant work in the literature cited?
- **Presentation (30%)**: Is the work presented in a clear, logical, and coherent manner? (including both oral presentation and final project report)
- **Novelty (20%)**: Is the work carried out novel? Does it advance the state of the art?

Each final project report is in the form of an ACL conference-style short paper. The main content of the paper is limited to 4 pages, with the references taking up a maximum of 1 additional page. You must use the ACL 2021 conference paper style files available at the following (latex or Microsoft WORD):

<https://www.overleaf.com/latex/templates/instructions-for-acl-ijcnlp-2021-proceedings/mhxffkjdwymb>
<https://2021.aclweb.org/downloads/acl-ijcnlp2021-templates.zip>

The page limit must be strictly adhered to. Marks will be deducted if the main content of your paper exceeds the 4-page limit, or the references exceed 1 page, or the font size is not Times Roman 11-point font.

Each project team must also submit the source code implemented in the project, all external (supporting) code and data, and instructions on how to run their code. Each team will give a 15-minute oral presentation in a video recording at the end of the course.
