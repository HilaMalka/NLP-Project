import project_evaluate as ev
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, Adafactor
from datasets import load_dataset, load_metric, DatasetInfo
from torch.utils.data import DataLoader, Dataset
from utils import flatten, add_dp
import numpy as np
from datetime import datetime
import spacy


#model_t5_path = "t5-small"
model_t5_path = "t5-base"
model_t5_big = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_t5_big)
metric = load_metric("sacrebleu")
now = datetime.now()
daytime = now.strftime("%d%m") + '_' + now.strftime('%H%M')


def remove_mod(sen):
    i = sen.find('Modifier')
    return sen[:i]


def create_data(data):
    file_en, file_de = data
    sentences = []
    for en_sen, de_sen in zip(file_en, file_de):
        sentences.append({'translation': {'en': en_sen, 'de': de_sen}})
    return sentences


class createDataset(Dataset):
    def __init__(self, input, target, tokenizer):
        self.max_len = 256
        # prefix = 'Deutsch ins Englische übersetzen'
        prefix = 'translate German to English: '
        self.input_data = [prefix + s for s in input]
        self.target_data = [t for t in target]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.tokenizer(self.input_data[idx], padding=True,
                           max_length=self.max_len,
                           truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(self.target_data[idx], padding=True,
                                    max_length=self.max_len,
                                    truncation=True)
        x["labels"] = labels.data["input_ids"]
        return x


def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    #print(preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print(decoded_preds[0])

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    # with open('results_' + daytime + '.txt', 'a') as f:
    #     f.write(str(result))
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    # print('blue is:', result['bleu'])
    return result


if __name__ == '__main__':
    source_lang = 'German'
    target_lang = 'English'

    # temp_train_data = ev.read_file('train.labeled')
    # temp_val_data = ev.read_file('val.labeled')
    # train_data = flatten(temp_train_data)
    # val_data = flatten(temp_val_data)

    model_name = 'Good_' + daytime
    train_data = add_dp(ev.read_file('train.labeled'))
    #val_data = add_dp(ev.read_file('val_small.labeled'))
    _, val_ger = ev.read_file('val.unlabeled')
    val_en, _ = ev.read_file('val.labeled')

    #ger_text = [remove_mod(ger) for ger in val_ger]
    val_data = (val_en, val_ger)
    # train_data = ev.read_file('train_small.labeled')
    # val_data = ev.read_file('val_small.labeled')

    train_dataset = createDataset(train_data[1], train_data[0], tokenizer)
    val_dataset = createDataset(val_data[1], val_data[0], tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_t5_path, output_attentions=True)
    #model = AutoModelForSeq2SeqLM.from_pretrained('good_model')
    batch_size = 4
    #model_name = model_t5_path

    args = Seq2SeqTrainingArguments(
        #f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        model_name,
        evaluation_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=7,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        generation_max_length=256,
        adafactor=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # optimizers=optimizer
    )
    trainer.train()
    trainer.save_model(model_name)
