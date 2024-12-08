# !pip install llama-cpp-python
# !pip3 install huggingface-hub>=0.17.1
# !huggingface-cli download TheBloke/Llama-2-13B-GGUF llama-2-13b.Q5_K_M.gguf --local-dir . --local-dir-use-symlinks False
# !pip install pandas
# !pip install pyinflect

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar
import pandas as pd
import json
import re
import numpy as np
from pyinflect import getAllInflections
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

llama_sense_path = 'output-systematic_outputs.json'
with open(llama_sense_path, 'r') as file:
    llama_senses = json.load(file)

llama_sense_dict = {}
llama_sense_dict = {key: None for key in llama_senses.keys()}
for verb in llama_sense_dict.keys():
  llama_sense_dict[verb] = llama_senses[verb]['systematic_outputs/output_0.7prompt6_poly_seeded.csv']

verb_list = []
sense_list = []

for verb, senses in llama_sense_dict.items():
    for sense in senses.values():
        verb_list.append(verb)
        sense_list.append(sense)

df = pd.DataFrame({'verb': verb_list, 'sense': sense_list})
df['sense'] = df['sense'].str.lower()
# df = df[df['sense'] != 'monosemous']
llama_senses_df = df.reset_index(drop=True)
llama_senses_df['id'] = llama_senses_df.groupby('verb').cumcount() + 1
llama_senses_df['id'] = llama_senses_df['id'].astype(str)
llama_senses_df['sense_id'] = llama_senses_df['verb'] + llama_senses_df['id']
llama_senses_df = llama_senses_df.drop(columns = ["id"])

def split_sense_id(sense_id):
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')
    match = pattern.match(sense_id)

    if match:
        result = (match.group(1), str(match.group(2)))
        verb = result[0]
        number = result[1]
    else:
        print("No match found.")
    return verb, number

for index, row in llama_senses_df.iterrows():
    sense_id = row['sense_id']
    split_sense = split_sense_id(sense_id)
    sense_descr = llama_sense_dict[split_sense[0]][split_sense[1]]
    sense_descr = sense_descr.lower()
    llama_senses_df.at[index, 'sense'] = sense_descr

mono_sense_df = pd.read_csv('mono_sense_descr.csv')
mono_sense_dict = mono_sense_df.set_index('verb')['sense_descr'].to_dict()

for index, row in llama_senses_df.iterrows():
    if row['verb'] in mono_sense_dict.keys():
        llama_senses_df.at[index, 'sense'] = mono_sense_dict[row['verb']]

df_duplicated_llama = pd.concat([llama_senses_df] * 10, ignore_index=True)
df_duplicated_llama['sentence_id'] = df_duplicated_llama['sense_id'] + '_' + (df_duplicated_llama.groupby('sense_id').cumcount() + 1).astype(str)

propbank_path = 'all_verbs_supermereo.csv'
propbank_df = pd.read_csv(propbank_path)
propbank_df['trans_sense_defs'] = propbank_df['trans_sense_defs'].str.strip('"')
propbank_df = propbank_df.drop(columns = ["Unnamed: 0"])

result_dict = {}

for index, row in propbank_df.iterrows():
    class_definitions = row['trans_sense_defs'].split(';')
    
    current_verb_dict = {}
    last_class_name = ""
    
    for definition in class_definitions:
        parts = definition.split(':')
        if len(parts) == 2:
            sense_name = parts[0].strip()
            class_description = parts[1].strip()
            
            if sense_name.startswith(row['verb'] + '.'):
                current_verb_dict[sense_name] = class_description
                last_class_name = sense_name
            else:
                current_verb_dict[last_class_name] += '; ' + sense_name + ': ' + class_description
    
    verb = row['verb']
    result_dict[verb] = current_verb_dict

propbank_sense_dict = result_dict.copy()

verbs = []
senses = []
sense_descrs = []

# Iterate through the outer dictionary
for verb, senses_dict in propbank_sense_dict.items():
    for sense, sense_descr in senses_dict.items():
        verbs.append(verb)
        senses.append(sense)
        sense_descrs.append(sense_descr)

propbank_senses_df = pd.DataFrame({'verb': verbs, 'sense_id': senses, 'sense': sense_descrs})
propbank_senses_df = pd.concat([propbank_senses_df] * 10, ignore_index=True)
propbank_senses_df['sentence_id'] = propbank_senses_df['sense_id'] + '_' + (propbank_senses_df.groupby('sense_id').cumcount() + 1).astype(str)

def get_past_tense(verb):
    inflections = getAllInflections(verb, 'V')
    return inflections['VBD'][0]

def calculate_surprisal(log_probs):
    log_probs_sum = -np.sum(log_probs)
    return log_probs_sum

def generate(template, initial_seed, type, init_temperature=0.8):
    final = template.copy()

    llm = Llama(model_path="llama-2-13b.Q5_K_M.gguf", logits_all=True)

    sentences = []
    for index, row in template.iterrows():
        skip = ['see.09_5', 'see.09_6', 'see.09_7', 'see.09_8', 'see.09_9', 'see.09_10']
        if row['sentence_id'] in skip:
            pass
        else:
            print(row['sentence_id'])
            seed = initial_seed + (int(row['sentence_id'][-1])-1)

            verb_root = row['verb']
            verb_past = get_past_tense(verb_root)
            verb_sense_gloss = row['sense']

            prompt = f'An example of a sentence containing the verb "{verb_root}" in the sense "{verb_sense_gloss}": '
            grammar = fr'''
                root ::= np v np "."
                np ::= d n
                d ::= "the "
                n ::= [a-z]+ " "
                v ::= "{verb_past} "
                '''
            temperature = init_temperature
            while True:  # Loop to generate a unique sentence with an increased seed
                seed_difference = seed - initial_seed
                temperature_increase_count = seed_difference // 100
                temperature = round(init_temperature + (0.1 * temperature_increase_count), 2)

                output = llm(
                    prompt=prompt,
                    max_tokens=32,
                    grammar=LlamaGrammar.from_string(grammar),
                    echo=False,
                    seed=seed,
                    logprobs=1,
                    temperature=temperature
                )

                sentence = output['choices'][0]['text']
                cleaned_string = sentence.rstrip('.')
                if cleaned_string not in sentences:
                    sentences.append(cleaned_string)
                    surp = calculate_surprisal(output['choices'][0]['logprobs']['token_logprobs'])

                    template.at[index, 'sentence'] = cleaned_string
                    template.at[index, 'seed'] = seed
                    template.at[index, 'surprisal'] = surp
                    template.at[index, 'temp'] = temperature

                    if type=="llama":
                        template.to_csv("llamasense_generation.csv", index=False)
                    elif type=="test":
                        template.to_csv("test.csv", index=False)
                    else:
                        template.to_csv("propbanksense_generation.csv")
                    break  # break loop if a unique sentence is generated
                else:
                    seed += 1  

    return template


columns = ["", "verb", "sense_id", "sense", "sentence_id", "sentence", "seed", "surprisal", "temp"]
data = [
    [0, "see", "see.09", "visit/consultation by medical professional", "see.09_1", pd.NA, pd.NA, pd.NA, pd.NA],
    [1, "see", "see.09", "visit/consultation by medical professional", "see.09_2", pd.NA, pd.NA, pd.NA, pd.NA],
    [2, "see", "see.09", "visit/consultation by medical professional", "see.09_3", pd.NA, pd.NA, pd.NA, pd.NA],
    [3, "see", "see.09", "visit/consultation by medical professional", "see.09_4", pd.NA, pd.NA, pd.NA, pd.NA],
    [4, "see", "see.09", "visit/consultation by medical professional", "see.09_5", pd.NA, pd.NA, pd.NA, pd.NA],
    [5, "see", "see.09", "visit/consultation by medical professional", "see.09_6", pd.NA, pd.NA, pd.NA, pd.NA],
    [6, "see", "see.09", "visit/consultation by medical professional", "see.09_7", pd.NA, pd.NA, pd.NA, pd.NA],
    [7, "see", "see.09", "visit/consultation by medical professional", "see.09_8", pd.NA, pd.NA, pd.NA, pd.NA],
    [8, "see", "see.09", "visit/consultation by medical professional", "see.09_9", pd.NA, pd.NA, pd.NA, pd.NA],
    [9, "see", "see.09", "visit/consultation by medical professional", "see.09_10", pd.NA, pd.NA, pd.NA, pd.NA]
]

# test
# test_df = pd.DataFrame(data, columns=columns)
# test_df_final = generate(test_df, 42, "test")

# output = generate(df_duplicated, 42)
output = generate(propbank_senses_df, 42, "propbank")

df = pd.read_csv('propbanksense_generation.csv')
df_sorted = df.sort_values(by='sense_id').reset_index(drop=True) #sort by sense_id
df_minimized = df_sorted.drop_duplicates(subset='sentence')
df_top_four = df_minimized.groupby('sense_id').apply(lambda x: x.nsmallest(4, 'surprisal')).reset_index(drop=True)

df_top_four.to_csv("propbanksense_generation_top_four.csv")
