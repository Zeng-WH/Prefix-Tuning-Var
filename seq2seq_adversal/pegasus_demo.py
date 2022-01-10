from transformers.data.processors.glue import OutputMode
from transformers import BartModel, BartForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

gpt2_tokenizer = AutoTokenizer.from_pretrained('./model/pegasus-large/pegasus-large')

data_token = 'long live chairman xi'

data_tokenize = gpt2_tokenizer(data_token, return_tensors='pt')
pad_token_id = gpt2_tokenizer.convert_tokens_to_ids("<pad>")

input_ids = data_tokenize['input_ids']

gpt2_config = AutoConfig.from_pretrained('./model/pegasus-large/pegasus-large')
gpt2_config.preseqlen = 200
gpt2_config.use_prefix = True
model_gpt2 = BartForConditionalGeneration.from_pretrained(
                './model/pegasus-large/pegasus-large',
                from_tf=bool(".ckpt" in './model/pegasus-large/pegasus-large'),
                config=gpt2_config,
            )

model_gpt2 = model_gpt2.cuda()

decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)

ouput = model_gpt2(input_ids.to(model_gpt2.device), decoder_input_ids = decoder_input_ids.to(model_gpt2.device), use_cache=True,
return_dict=True)



a = ouput.past_key_values


output_list = []

for item in a:
    output_list.append(item['encoder_decoder']['prev_key'])
    output_list.append(item['encoder_decoder']['prev_value'])
    b = item['encoder_decoder']['prev_key']
    print(b.shape)




#print(a)

#print(a.shape)

print("bupt")