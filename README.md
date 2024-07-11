# NLP-Project
Translation model for translating paragraphs from German to English

In our project, we tackled the task of translating between German and English using pretrained T5 models. Initially, we calibrated the process using the T5-small model. Subsequently, we upgraded to the T5-base model and incorporated a tokenizer trained on the T5-large model, which performed better than the base model but was computationally prohibitive to utilize directly.

General Algorithm Explanation:

We added a prefix requesting translation from German. We experimented with writing "translate from German to English" in both languages, finding better performance with the English version.

We endeavored to find optimal training parameters, including batch size, learning rate, weight_decay, and max_len_generated. Various parameter combinations were tested to gauge performance. Ultimately, we settled on:

We attempted creative thinking about the task and utilized all available resources within the exercise constraints. Our main efforts included:

Attempting to expand the dataset by segmenting the provided paragraphs to enhance training item information. This effort neither improved nor impaired performance. Ultimately, we abandoned this approach due to the inconsistent order among sentence segments in translated paragraphs, which hindered model learning. Additionally, the generated sentences were significantly shorter than those required in validation and test files.

Utilizing additional information from unlabeled files.

We opted to utilize only roots by using the spacy library, capable of computing dependency trees to provide information for training files and enhance the model. We received data from the library and added it to the model's input. Performance improved significantly.

We did not succeed in integrating modifiers due to their varying and distinct quantities between sentences and different roots.

Expected Accuracy Percentage: On validation, we achieved a BLEU score of approximately 40. We anticipate a similar score on the competition file, assuming domain and sentence length consistency, with a variation of +/-1.
