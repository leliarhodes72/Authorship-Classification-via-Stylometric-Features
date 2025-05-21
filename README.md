---
title: "Authorship Classification via Stylometric Features"
excerpt: "A classification pipeline that uses TF-IDF, cosine similarity, and gradient boosting to determine whether two text spans share an author."
---

This project involved building a sentiment- and style-based classification system to determine whether two spans of English text were written by the same author. The task—part of a digital stylometry challenge—was a binary classification problem requiring the model to assign either label `0` (different authors) or `1` (same author).

## Task Summary

The objective of the task was to identify whether two pieces of English text were written by the same person. This type of task is known as **authorship identification** and has applications in forensic linguistics, authorship attribution in literary analysis, and online behavior tracking. The challenge was structured as a binary classification problem, where the input consisted of two spans of text separated by a delimiter (`[SNIPPET]`), and the model had to predict whether the spans were authored by the same individual (label `1`) or not (label `0`).

A key complexity of this task was the absence of author labels—i.e., no metadata about authorship beyond the span pairs themselves. Therefore, the model needed to rely entirely on linguistic features and patterns in the text to infer authorship, which requires both semantic and stylistic analysis. The ability to distinguish writing styles without topic-level cues was central to the success of this project, and it presented a uniquely challenging opportunity to apply a wide range of NLP tools and evaluation strategies.

## Exploratory Data Analysis

The training data contained 1601 samples. Each sample had a single text field that consisted of two concatenated spans of English text, separated by the `[SNIPPET]` token. These spans varied in length and structure but generally ranged from 2 to 4 sentences each. The data was highly imbalanced:

- **Label 0 (Different authors):** 1,245 samples
- **Label 1 (Same author):** 356 samples

This imbalance presented a challenge for training, particularly for recall on the minority class. In order to address this, I later incorporated class-balancing techniques during model development.

An initial inspection of text pairs revealed substantial stylistic variation between authors. However, many pairs that were semantically similar—due to common topics, genres, or even quotation—did not share authorship. As a result, naive semantic similarity was not sufficient for classification, and deeper stylistic analysis was necessary.

One illustrative example of a text pair from the dataset was:
> Span 1: “When his speaker remained silent Dirrul assumed he had been understood. He began to feel the pull of Vininese gravity, found himself in trouble with his ship. He tried to keep the disabled cargo carrier relatively stationary, so that the Vininese repair ships could locate him. With only one power tube, however, maneuver was impossible. The battered ship plunged out of control toward the planet. For an hour Dirrul fought with all the skill he knew. A thousand feet above the surface he managed to force the ship to level off temporarily. [SNIPPET] How big is a spation in space?"" ""Van Manderpootz has even measured that. A chronon is the length of time it takes one quantum of energy to push one electron from one electronic orbit to the next. There can obviously be no shorter interval of time, since an electron is the smallest unit of matter and the quantum the smallest unit of energy.”
> 
> Span 2: “"She stiffened. "" If it was as simple as insanity, I would."" ""Please, Beth."" He wandered to the fireplace and threw in more wood. ""The stealing does bother me. I think the changing is good. [SNIPPET] I won't talk to anybody."" Beth moved to the window. The wind had died. "" I don't know, Ben."" ""Let it rest. Let's have the drink."" He came to her side."”

While these spans clearly relate thematically, the model had to decide whether these similarities indicated a shared author or not. This example also illustrates the difficulty of separating **topic similarity** from **stylometric consistency**—a distinction central to authorship attribution tasks.

## Preprocessing and Feature Engineering

The preprocessing pipeline was designed to clean and normalize the input text. I used the following steps:

1. **Lowercasing**: Convert all text to lowercase to reduce vocabulary size.
2. **Punctuation removal**: Strip punctuation using Python’s `string.punctuation`.
3. **Tokenization**: Tokenize each span using NLTK’s `TreebankWordTokenizer`.
4. **Stopword removal**: Remove common English stopwords.
5. **Lemmatization**: Lemmatize all tokens using NLTK’s `WordNetLemmatizer`.

These steps allowed the model to focus on meaningful content words while reducing sparsity in the feature set. The text was then split into two columns—`SPAN1` and `SPAN2`—based on the `[SNIPPET]` delimiter. Each span was preprocessed independently, and I used both the individual spans and their combination to build features.

## Feature Extraction

To convert text into numerical representations, I used the **TF-IDF vectorizer** from `scikit-learn`, configured with the following parameters:

- `ngram_range=(1, 2)`: Included unigrams and bigrams to capture short phrases
- `max_features=5000`: Capped feature size to reduce sparsity and overfitting
- `min_df=2`: Ignored terms that appeared only once
- `max_df=0.8`: Removed overly common terms that might dilute specificity

TF-IDF vectors were generated separately for each span and their combination. To enhance performance, I also calculated the **cosine similarity** between the two spans’ TF-IDF vectors. This additional feature captured overall semantic similarity and was particularly helpful for distinguishing paraphrased content or style continuity. The cosine similarity feature was especially helpful when two spans conveyed similar meanings with different surface forms, allowing the model to capture stylistic coherence.

The final feature matrix was constructed by horizontally stacking the TF-IDF vectors of the combined span and the cosine similarity values using `scipy.sparse.hstack`. This ensured efficient storage and integration into downstream modeling pipelines.

## Modeling Approach

Initially, I used **Logistic Regression** as a baseline model due to its speed and interpretability. To address the class imbalance, I used the `class_weight='balanced'` option, which penalizes misclassifications from the minority class more heavily.

Once the baseline was established, I implemented a **Gradient Boosting Classifier (GBC)** to improve performance. GBC is an ensemble method that builds sequential decision trees, each one trained to correct the errors of its predecessors. This allowed the model to capture nonlinear relationships and interactions between features.

The full pipeline included:
- Preprocessing and tokenization
- Feature engineering with TF-IDF and cosine similarity
- Classifier training with GBC
- Evaluation on test data

I trained and validated my model using cross-validation and final test submission. During the experimentation phase, I adjusted hyperparameters such as learning rate, number of estimators, and max depth. However, due to the relatively small size of the dataset, I prioritized model simplicity and feature quality over extensive hyperparameter tuning.

## Evaluation and Results

Model performance was evaluated through a blind test submission. The final Gradient Boosting model achieved a **leaderboard score of 0.57992**, which represented a **7.09% improvement over the baseline** Logistic Regression model. **This score reflects the macro-averaged F1 metric**, which balances precision and recall across both classes. For context, the top-performing model in the competition scored approximately **0.621** using an ensemble of fine-tuned transformer models (e.g., RoBERTa). While my model prioritized lightweight, interpretable techniques, it still performed competitively and showcased the effectiveness of feature engineering without relying on large-scale pretrained architectures.

This result demonstrated that combining stylistic features with semantic similarity, and employing a non-linear model, led to more accurate authorship classification. Although the score reflects a moderately difficult task, it also highlighted the importance of feature engineering and model selection in text classification.

Additionally, the project reinforced the value of carefully constructed baselines and incremental improvements. Each modeling decision—from preprocessing to feature choice—had a clear and measurable impact on overall performance.

## Error Analysis

Misclassifications were primarily caused by:
- **Semantic similarity without shared authorship**: Some pairs were topically or stylistically alike, which misled the cosine similarity metric.
- **Vocabulary mismatch**: Rare or domain-specific words increased feature sparsity.
- **Class imbalance**: The underrepresentation of the positive class affected recall.

Example errors included cases where span 1 used vivid, poetic language and span 2 was more direct or narrative in tone, despite thematic overlap. In such cases, the classifier often defaulted to predicting different authors.

Future improvements could include:
- **Using contextual embeddings** such as BERT or RoBERTa for richer semantic features
- **Augmenting data** through paraphrasing or synthetic text generation
- **Applying sampling techniques** like SMOTE to balance classes
- **Exploring stylometric markers** such as sentence length, part-of-speech tags, and punctuation frequency

These enhancements could help the model better generalize to unseen authors and edge cases where surface-level features fail.

## Learning Outcomes

This project allowed me to demonstrate the following learning outcomes of the MS-HLT program:

1. **Programming proficiency**: I wrote clean, modular code in Python using libraries like `nltk`, `scikit-learn`, and `pandas`.
2. **Applied NLP knowledge**: I applied tokenization, lemmatization, TF-IDF vectorization, and cosine similarity to real-world text data.
3. **Tool integration**: I combined multiple tools and packages into a coherent ML pipeline.
4. **Analytical thinking**: I conducted exploratory data analysis, tuned model parameters, and performed error analysis to guide improvements.
5. **Reproducibility and professionalism**: The project was organized for easy sharing and reproducibility via GitHub, and my code was structured to support collaboration and version control.
