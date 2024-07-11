from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import project_evaluate as ev
from tqdm import tqdm
from main import *
from utils import add_dp
model_t5_path = "t5-large"
#tokenizer = T5Tokenizer.from_pretrained(model_t5_path)

#model_path = 'good_model_2/checkpoint-6000'
model_path = 'comp_model/checkpoint-12500'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
batch_size = 32

args = Seq2SeqTrainingArguments(
    f"Prediction",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    generation_max_length=256
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("**********************")


def remove_mod(sen):
    i = sen.find('Modifier')
    return sen[:i]


def remove_root(sen):
    i = sen.find('Roots in English')
    return sen[:i]


def generate_file(file, name):
    _, ger_text = ev.read_file(file)
    #ger_text = [remove_mod(ger) for ger in ger_text]
    test_en = [''] * len(ger_text)
    test_dataset = createDataset(ger_text, test_en, tokenizer)
    prediction = trainer.predict(test_dataset=test_dataset)[0]
    en_pred = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    ger_text = [remove_root(ger) for ger in ger_text]

    with open(name, 'w') as w:
        for en_pred, ger in zip(en_pred, ger_text):
            w.write('German: \n')
            w.write(ger)
            w.write('\n')
            w.write('English: \n')
            w.write(en_pred)
            w.write('\n')
            w.write('\n')


def generate_val():
    generate_file('val.unlabeled', 'val_208602557_313312753.labeled')


def generate_comp():
    generate_file('comp.unlabeled', 'comp_208602557_313312753.labeled')


if __name__ == '__main__':
    print("Generating Validation file")
    generate_val()
    print("Generating Comp file")
    generate_comp()

    # #dest_name = 'val_208602557.labeled'
    # dest_name = 'x.labeled'
    # generate_file(trainer, 'val.unlabeled', dest_name)
    # #generate_file(trainer, 'comp.unlabeled', 'com')
    # ev.calculate_score('val.labeled', dest_name)




