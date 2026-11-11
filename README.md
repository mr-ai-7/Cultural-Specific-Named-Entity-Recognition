# Cultural-Specific Named Entity Recognition

This IIT Hyderabad NLP project uses BERT to identify Indian cultural entities in English sentences, including foods, festivals, languages, dances, and clothing. For example, in “Gondi is part of Indian tradition,” the goal is to recognize “Gondi” as a language.

 [View the presentation](presentation.pdf)

## Dataset and approach

1. Map cultural concepts from the [DIWALI dataset](https://huggingface.co/datasets/nlip/DIWALI) to entity categories.
2. Use sentence templates to generate 12,000 labeled examples. Each token receives a BIO tag marking the beginning of an entity (`B`), a continuation (`I`), or a word outside an entity (`O`).
3. Split the examples into 9,600 for training, 1,200 for validation, and 1,200 for testing.
4. Fine-tune `bert-base-uncased` for token classification and evaluate entity-level F1 using `seqeval`.

## Results

After five training epochs, the model achieved an **entity-level F1 score of 0.8649** on the generated test set. The split was made by sentence, so the same cultural concept or template can appear in both training and testing. Performance on independently written text remains unmeasured.

## Run the project

Open `cultural_ner.ipynb` in Google Colab or Jupyter. Run the following command in a new cell at the top of the notebook:

    %pip install torch transformers datasets accelerate seqeval scikit-learn numpy matplotlib

Run the remaining cells in order. The notebook downloads the dataset and base model, generates the labeled sentences, trains BERT, and evaluates it. A GPU is recommended for training.

The notebook saves the generated data to `csi_ner_dataset_final.json` and the trained model and tokenizer to `final_model_best/`. Random template selection can produce a slightly different dataset and score on another run.
