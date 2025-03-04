# Task 2: Labeling Dataset in CoNLL Format for Named Entity Recognition (NER)

## Overview
This task involves labeling a portion of the dataset in **CoNLL format** to train a Named Entity Recognition (NER) model. The primary goal is to annotate key entities, such as **products, prices, and locations**, in Amharic text from Ethiopian-based Telegram e-commerce channels.

## Dataset
- **Source**: Preprocessed messages from Telegram e-commerce channels.
- **Format**: Text-based dataset containing product listings, descriptions, prices, and locations.
- **Language**: Amharic.

## Steps

### 1. Data Preparation
- Extract relevant text messages from the preprocessed dataset.
- Normalize the text (handling special characters, spacing, and Amharic linguistic variations).
- Tokenize the text into words.

### 2. Entity Labeling
- Label tokens using the **BIO tagging** scheme:
  - `B-Product` (Beginning of a product name)
  - `I-Product` (Inside a product name)
  - `B-Price` (Beginning of a price mention)
  - `I-Price` (Inside a price mention)
  - `B-Location` (Beginning of a location name)
  - `I-Location` (Inside a location name)
  - `O` (Other words that do not belong to any entity)

### 3. Formatting in CoNLL Style
Each token should be annotated in the following format: