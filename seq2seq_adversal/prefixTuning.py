# from transformers import Trainer
import torch
from torch._C import dtype, set_flush_denormal
from torch.nn.functional import embedding
#from torch._C import T, set_flush_denormal
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, PretrainedBartModel, modeling_gpt2
from torch import BoolStorage, nn

## add low data setting , write by Andrew Zeng, start it
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.generation_utils import set_scores_to_inf_for_banned_tokens
from transformers.modeling_bart import shift_tokens_right


from transformers import PegasusForConditionalGeneration

from transformers import AutoConfig, AutoTokenizer

from gradient_reversal import GradientReversal
import math
## add low data setting , write by Andrew Zeng, end it


def domain_w2v(domain_words, gpt_model, gpt_tokenizer):

    embedding_list = []
    for word in domain_words:
        inputs = gpt_tokenizer(word, return_tensors="pt")

        #inputs = inputs.cuda()

        #print(self.forzen_gpt2.device)

        inputs = inputs.to(gpt_model.device)

        #print(inputs)

        outputs_hidden_states = gpt_model(**inputs, return_dict=True).last_hidden_state


        #print(outputs_hidden_states.shape)

        embedding_list.append(outputs_hidden_states)

    return embedding_list


def cal_dist(domain_embedding, forzen_hiddenstates, bsz, dist_list):
    for index, item in enumerate(domain_embedding):
        item = item.repeat(bsz, 1, 1)


        bsz_dst = torch.cdist(item, forzen_hiddenstates)

        bsz_dst = torch.min(torch.mean(bsz_dst, 2), 1, True).values


        dist_list.append(bsz_dst)
    return dist_list



class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        #self.shared = model_gpt2.model.shared



        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        deep_param = self.use_deep

        
        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

        self.lowdata_token = False
        self.lowdata_token = None


        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4

        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print('the is just for baseline checking!!! We reinitialize the LM embeddings and try cat '
                          'and peek.')
                    print('BASELINE'*100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)



            if self.mode_para == 1:
                print('mode_para=1, for dataless.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print('mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes...')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3


            elif self.mode_para == 3:
                print('mode_para=3, OLD VERSION: many parameters.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.preseqlen * config.n_layer * 2 * config.n_embd), nn.Tanh())
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p1_infix
                else:
                    self.get_prompt = self.get_prompt_p1
        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))


            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                if low_data_init == 1:
                    print('IN THE LOW DATA SETTING, EXPLORE INITIALIZATION FOR DIRECT OPTIM...')
                    # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p22
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
                    self.control_trans = self.lowdata_init_train1(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
                    print(self.control_trans.shape)
                elif low_data_init == 2:
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first')
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5

                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    # sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)

                elif low_data_init == 3:
                    self.use_encoder_prefix = True
                    self.use_cross_prefix = True
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    input_ids = tokenizer(self.lowdata_token, return_tensors='pt')["input_ids"]
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    self.preseqlen = len(input_ids)
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                          'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                    if self.use_encoder_prefix:
                        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                        self.control_trans_enc = nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                    if self.use_cross_prefix:
                        self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                        self.control_trans2 = nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                    self.get_prompt = self.get_prompt_domain






            # DIFFERENT PARAMETRIZATION:
            elif not deep_param:
                low_data_init = 0
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)



                    self.wte_dst = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)

                    '''
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                    '''
                    
                    

                    #self.control_trans_enc = nn.RNN(self.n_embd, self.n_embd * self.match_n_layer, batch_first=True, bidirectional=True).to(self.device)

                    self.control_trans_enc = []


                    self.control_trans_enc.append(nn.LSTM(self.n_embd, self.n_embd, bidirectional=True, batch_first=True))
                    for item in range(self.match_n_layer - 1):
                        self.control_trans_enc.append(nn.LSTM(self.n_embd * 2, self.n_embd, bidirectional=True, batch_first=True))



                    self.reverse_net = nn.Sequential(
                        nn.Linear(self.match_n_layer * 2 *self.match_n_head*self.match_n_embd, self.match_n_embd* self.match_n_layer * 2),

                        nn.ReLU(),

                        nn.Linear(self.match_n_embd* self.match_n_layer * 2, self.match_n_embd),

                        GradientReversal(alpha=1.)
                    )

                    self.domain_classifier = nn.Sequential(

                        nn.Linear(self.match_n_embd, math.floor(self.match_n_embd / 2)),

                        nn.ReLU(),

                        nn.Linear(math.floor(self.match_n_embd / 2), 2)
                    )


                    
                    #self.control_trans_enc.cuda(self.device)

                    self.control_trans_enc_dynamic = []

                    self.control_trans_enc_dynamic.append(nn.LSTM(self.n_embd, self.n_embd, bidirectional=True, num_layers=1, batch_first=True))
                    for item in range(self.match_n_layer - 1):
                        self.control_trans_enc_dynamic.append(nn.LSTM(self.n_embd * 2, self.n_embd, bidirectional=True, num_layers=1, batch_first=True))



                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))



            else:
                low_data_init = 0
                print('UNDER PARAMETRIZATION DEEP 1')

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5


                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))

        if low_data_init == 2:
            self.lowdata_init_train2(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
        elif low_data_init == 3:
            print('use pt for this tensor', self.lowdata_token)
            #self.lowdata_init_train3(gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token))
            self.init_domain_train(gpt2=model_gpt2, domain_word=self.lowdata_token)

        self.forzen_gpt2_config = AutoConfig.from_pretrained('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/model/pegasus-large/pegasus-large')
        self.forzen_gpt2_config.preseqlen = 200
        self.forzen_gpt2_config.use_prefix = True
        self.forzen_gpt2_tokenizer = AutoTokenizer.from_pretrained('/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/model/pegasus-large/pegasus-large')

        self.pad_token_id = self.forzen_gpt2_tokenizer.convert_tokens_to_ids("<pad>")

        #self.forzen_gpt2_tokenizer = model_gpt2_tokenizer


        '''

        self.forzen_gpt2 = PegasusForConditionalGeneration.from_pretrained(
            '/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/model/pegasus-large/pegasus-large',
            from_tf=bool(".ckpt" in '/home/hadoop-aipnlp/cephfs/data/zengweihao02/NAACL2021/Prefix-Tuning/seq2seq/model/pegasus-large/pegasus-large'),
            config=self.forzen_gpt2_config,
        )

        for name, param in self.forzen_gpt2.named_parameters():
            param.requires_grad = False



        '''

        '''

        for name, param in self.shared.named_parameters():
            param.requires_grad = False
        '''

        #self.forzen_gpt2 = self.forzen_gpt2.cuda()

        model_gpt2 = model_gpt2.cuda()


        self.train_domain_words = "train on reference book table time cambridge be 15 day will me people it yes great was all reservation from 00 should looking thanks booking leave am centre one do no by 30 with travel minutes 45 saturday try leaves leaving 17 booked 12 price"

        self.taxi_domain_words = "taxi contact from leave by arrive be up where will car after time booked destination departing going leaving want what 15 pick great go 00 45 picked when 30 thanks book day completed grey type take today booking white black yellow do welcome ok red"

        self.hotel_domain_words = "hotel book free on reference yes be stay people do parking price range table great was centre looking day one house area star nights reservation guesthouse thanks wifi booking try place will am how expensive any or cheap about find town starting north another what"

        self.attraction_domain_words = "entrance fee postcode museum sock missing attraction phone their college visit museums express information town holiday inn attractions centre entertainment go colleges corner finders fun church architecture pounds 812660 corn exchange pool nightclub kettle street yard address cambridge today art tell dojo nightclubs castle thanks"

        self.restaurant_domain_words = "food phone town address part goodbye their serves restaurants 01223 price range south expensive centre italian located what cheap postcode good city indian east bye west north priced road an moderately looking chinese street serving european area type serve any about cherry hinton am find"

        self.train_domain_words = self.train_domain_words.split(" ")

        self.taxi_domain_words =  self.taxi_domain_words.split(" ")

        self.hotel_domain_words = self.hotel_domain_words.split(" ")

        self.attraction_domain_words = self.attraction_domain_words.split(" ")

        self.restaurant_domain_words = self.restaurant_domain_words.split(" ")


        self.all_domain_words = []

        self.all_domain_words.extend(self.train_domain_words)
        self.all_domain_words.extend(self.taxi_domain_words)
        self.all_domain_words.extend(self.hotel_domain_words)
        self.all_domain_words.extend(self.attraction_domain_words)
        self.all_domain_words.extend(self.restaurant_domain_words)

        #print(self.train_domain_words)


        self.train_domain_embeding = domain_w2v(self.train_domain_words, model_gpt2.model, self.forzen_gpt2_tokenizer)

        self.taxi_domain_embeding = domain_w2v(self.taxi_domain_words, model_gpt2.model, self.forzen_gpt2_tokenizer)

        self.hotel_domain_embedding = domain_w2v(self.hotel_domain_words, model_gpt2.model, self.forzen_gpt2_tokenizer)

        self.attraction_domain_embedding = domain_w2v(self.attraction_domain_words, model_gpt2.model, self.forzen_gpt2_tokenizer)

        self.restaurant_domain_embedding = domain_w2v(self.restaurant_domain_words, model_gpt2.model, self.forzen_gpt2_tokenizer)
        '''
        for word in self.train_domain_words:
            inputs = self.forzen_gpt2_tokenizer(word, return_tensors="pt")

            #inputs = inputs.cuda()

            #print(self.forzen_gpt2.device)

            inputs = inputs.to(self.forzen_gpt2.device)

            outputs_hidden_states = self.forzen_gpt2.model(**inputs)[0]

            print(outputs_hidden_states[0].shape)
            print(len(outputs_hidden_states))

            print(outputs_hidden_states[1].shape)

        '''

    








        



    def lowdata_init_train1(self, gpt2, tokenizer, sample_input):
        input = tokenizer(sample_input, return_tensors='pt')
        output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
        output = output.past_key_values
        print(len(output), output[0].shape)
        output = torch.cat(output, dim=0).detach()
        return torch.nn.Parameter(output)

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_train2(self, gpt2, tokenizer, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            input = tokenizer(sample_input, return_tensors='pt')
            output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()

        return

    def init_domain_train(self, gpt2, domain_word, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.input_tokenize = tokenizer(self.lowdata_token, return_tensors='pt')
        pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")

        input_ids = self.input_tokenize['input_ids']

        if isinstance(self, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(input_ids)
        else:
            decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
        with torch.no_grad():
            output = gpt2(input_ids.to(gpt2.device), decoder_input_ids=decoder_input_ids.to(gpt2.device), use_cache=True,
                          return_dict=True)
            output = output.past_key_values
            output_list = []
            output_list1 = []
            for item in output:
                output_list.append(item['self']['prev_key'])
                output_list.append(item['self']['prev_value'])
                output_list1.append(item['encoder_decoder']['prev_key'])
                output_list1.append(item['encoder_decoder']['prev_value'])
            output = torch.cat(output_list, dim=0)
            output1 = torch.cat(output_list1, dim=0)


        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt_list = []
            our_prompt_list1 = []
            our_prompt = self.get_prompt_domain(bsz=1)
            for item in our_prompt:
                our_prompt_list.append(item['self']['prev_key'])
                our_prompt_list.append(item['self']['prev_value'])
                our_prompt_list1.append(item['encoder_decoder']['prev_key'])
                our_prompt_list1.append(item['encoder_decoder']['prev_value'])

            our_prompt = torch.cat(our_prompt_list, dim=0)
            our_prompt1 = torch.cat(our_prompt_list1, dim=0)

            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            loss1 = loss_metrics(our_prompt1.to(gpt2.device), output1)
            total_loss = loss + loss1


            total_loss.backward()

            optimizer_temp.step()
            self.control_trans.zero_grad()

        return


    def lowdata_init_train3(self, gpt2, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(sample_input.to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(1, self.preseqlen,  self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p3_infix(self, src, control_code=None, gpt2=None, bsz=None):
        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        src_out = gpt2(input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True)
        src_repr = src_out.hidden_states[-1] #bsz, seqlen, hidden
        src_past_key_vals = src_out.past_key_values
        past_key_values = self.control_trans(src_repr) #bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        # print(past_key_values.shape)
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3))

        return full_lst

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values

    def get_prompt_domain(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2) # self.match_n_layer * 2, bsz_enc, self.match_n_head, seqlen, self.match_n_embd

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool()
                                  # bsz, preseqlen
                                  },
                         }

            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)
        return result

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1, h0_list = None, dst_ids = None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # self.match_n_layer * 2, bsz_enc, self.match_n_head, seqlen, self.match_n_embd



        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)
            past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)


        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)

            if dst_ids is not None:
                temp_control_enc_dyc = self.wte_dst(dst_ids)
                past_key_values_temp_dyc = temp_control_enc_dyc


            past_key_values_enc = []

            past_key_values_enc_dyc = []
            past_key_values_temp = temp_control_enc
            #past_key_values_temp_dyc = temp_control_enc_dyc
            #print(self.control_trans_enc)
            for item in range(self.match_n_layer):
                #past_key_values_temp = past_key_values_temp.cuda(self.device)
                control_trans_enc_temp = self.control_trans_enc[item]
                control_trans_enc_temp = control_trans_enc_temp.to(self.device)
                if h0_list is not None:
                    h0 = h0_list[item].to(self.device).contiguous()
                    c0 = torch.zeros(h0.shape).to(self.device).contiguous()
                    past_key_values_temp = control_trans_enc_temp(past_key_values_temp, (h0, c0))[0]

                else:
                    past_key_values_temp = control_trans_enc_temp(past_key_values_temp)[0]

                
                control_trans_enc_temp_dyc = self.control_trans_enc_dynamic[item]
                control_trans_enc_temp_dyc = control_trans_enc_temp_dyc.to(self.device)
                if dst_ids is not None:
                    past_key_values_temp_dyc = control_trans_enc_temp_dyc(past_key_values_temp_dyc)[0]



                #if dst_ids is not None:
                #    dst_embed = self.shared(dst_ids)

                #print(past_key_values_temp.shape)
                #h0 = h0_list[item]
                #c0 = torch.zeros(h0.shape)
                #past_key_values_temp = control_trans_enc_temp(past_key_values_temp, (h0, c0))[0]
                past_key_values_enc.append(past_key_values_temp)

                if dst_ids is not None:

                    past_key_values_enc_dyc.append(past_key_values_temp_dyc)

                #past_key_values_enc.append(self.control_trans_enc[item](temp_control_enc).cuda(self.device))
            #past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            past_key_values_enc = torch.cat(past_key_values_enc, 2)

    
            #Reverse_Net = nn.Sequential(nn.Linear())
    
            reverse_value = self.reverse_net(past_key_values_enc)
            
            domain_value = self.domain_classifier(reverse_value)[:, 0, :]


            #print("*********")

            #print(domain_value.shape)

            

            if dst_ids is not None:
                past_key_values_enc_dyc = torch.cat(past_key_values_enc_dyc, 2)

                past_key_values_enc = torch.cat((past_key_values_enc, past_key_values_enc_dyc), 1)
                #print(past_key_values_enc_dyc.shape)

            #print(past_key_values_enc.shape)
            bsz_enc, enc_seqlen, _ = past_key_values_enc.shape

            #self.preseqlen = seqlen
            #print("****************")
            #print(past_key_values_enc.shape)
            #print(self.match_n_layer * 2 * self.n_embd)
            past_key_values_enc = past_key_values_enc.view(bsz_enc, enc_seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2) # self.match_n_layer * 2, bsz_enc, self.match_n_head, seqlen, self.match_n_embd

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, enc_seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)

        return result, domain_value

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        input_embs = self.input_embs.to(self.device)
        past_key_values = self.control_trans(input_embs).expand(bsz, -1, -1) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control).mean(1).unsqueeze(1) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p1(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:

            if type(control_code) is tuple :
                assert False, 'Tuples'
                control_embs, control_word = control_code
                past_key_values = self.control_trans(control_embs)
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen * self.preseqlen, self.match_n_layer * 2,
                                                       self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                print(control_word, control_embs.shape)
            else:
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if not self.embMatch:
                    if self.wte:
                        temp_control = self.wte(control_code)
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(control_code)
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:
                    temp_control = control_code
                    # print(control_code.shape)
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen*self.preseqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward(self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # labels=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        oracle_ids=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        #print(oracle_ids)

        #print(input_ids.shape)

        #print(self.preseqlen)



        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        '''
        forzen_gpt2_config = AutoConfig.from_pretrained('./model/pegasus-large/pegasus-large')
        forzen_gpt2_config.preseqlen = 200
        forzen_gpt2_config.use_prefix = True
        forzen_gpt2_tokenizer = AutoTokenizer.from_pretrained('./model/pegasus-large/pegasus-large')

        pad_token_id = forzen_gpt2_tokenizer.convert_tokens_to_ids("<pad>")

        forzen_gpt2 = PegasusForConditionalGeneration.from_pretrained(
            './model/pegasus-large/pegasus-large',
            from_tf=bool(".ckpt" in './model/pegasus-large/pegasus-large'),
            config=forzen_gpt2_config,
        )

        forzen_gpt2 = forzen_gpt2.cuda(self.device)
        '''

        '''
        forzen_decoder_input_ids = shift_tokens_right(input_ids, self.pad_token_id)

        forzen_output = self.forzen_gpt2(input_ids.to(self.forzen_gpt2.device), decoder_input_ids=forzen_decoder_input_ids.to(self.forzen_gpt2.device), use_cache=True,
        return_dict=True)

        forzen_output = forzen_output.past_key_values

        output_list = []

        for item in forzen_output:
            h_0_prev_key  = item['encoder_decoder']['prev_key']
            #a.permute([])
            h_0_prev_value = item['encoder_decoder']['prev_value']
            #h_0_prev_key = h_0_prev_key[:, :, -1, :]


            #print(h_0_prev_key.shape)

            h_0_prev_key = torch.mean(h_0_prev_key, 2)
            #h_0_prev_value = h_0_prev_value[:, :, -1, :]

            h_0_prev_value = torch.mean(h_0_prev_value, 2)

            h_0_prev_key = h_0_prev_key.view(bsz, 1, -1)
            h_0_prev_value = h_0_prev_value.view(bsz, 1, -1)


            h_0 = torch.cat([h_0_prev_key, h_0_prev_value], 1)

            #print(h_0.shape)
            h_0 = h_0.permute([1, 0, 2])


            #print(h_0.shape)

            output_list.append(h_0)

        '''

        #print(len(output_list))

        forzen_hiddensates = gpt2_model.model(input_ids.to(gpt2_model.device), return_dict = True).last_hidden_state

        #print(forzen_hiddensates)






        dist_list = []

        '''

        for index, item in enumerate(self.taxi_domain_embeding):
            item = item.repeat(bsz, 1, 1)


            bsz_dst = torch.cdist(item, forzen_hiddensates)

            bsz_dst = torch.mean(torch.min(bsz_dst, 2).values, 1, True)


            dist_list.append(bsz_dst)
        '''


        dist_list = cal_dist(self.train_domain_embeding, forzen_hiddensates, bsz, dist_list)
        dist_list = cal_dist(self.taxi_domain_embeding, forzen_hiddensates, bsz, dist_list)
        dist_list = cal_dist(self.hotel_domain_embedding, forzen_hiddensates, bsz, dist_list)
        dist_list = cal_dist(self.attraction_domain_embedding, forzen_hiddensates, bsz, dist_list)
        dist_list = cal_dist(self.restaurant_domain_embedding, forzen_hiddensates, bsz, dist_list)


        dist_list=torch.cat(dist_list, 1)

        #print(dist_list.shape)


        top20 = torch.topk(dist_list, 20, dim=1)

        top20_indices = top20.indices

        #top20_indices = top20_indices.tolist()

        #print(top20_indices)

        #output_list_new = []

        #print(self.all_domain_words)
        input_token_list = []
        for i in range(top20_indices.shape[0]):
            input_token = []
            for j in range(top20_indices.shape[1]):

                #print(top20_indices[i,j])
                input_token.append(self.all_domain_words[top20_indices[i, j]])
            input_token = " ".join(input_token)
            #print(input_token)

            input_token_list.append(input_token)

        #print(input_token_list)

        input_ids_temp = self.forzen_gpt2_tokenizer(input_token_list, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        forzen_decoder_input_ids = shift_tokens_right(input_ids_temp, self.pad_token_id)

        forzen_output = gpt2_model(input_ids_temp.to(gpt2_model.device), decoder_input_ids=forzen_decoder_input_ids.to(gpt2_model.device), use_cache=True,
            return_dict=True)

        forzen_output = forzen_output.past_key_values

        output_list_new = []

        for item in forzen_output:
            h_0_prev_key  = item['encoder_decoder']['prev_key']


            #print(h_0_prev_key.shape)
                #a.permute([])
            h_0_prev_value = item['encoder_decoder']['prev_value']
                #h_0_prev_key = h_0_prev_key[:, :, -1, :]


                #print(h_0_prev_key.shape)

            h_0_prev_key = torch.mean(h_0_prev_key, 2)
                #h_0_prev_value = h_0_prev_value[:, :, -1, :]

            h_0_prev_value = torch.mean(h_0_prev_value, 2)

            h_0_prev_key = h_0_prev_key.view(bsz, 1, -1)
            h_0_prev_value = h_0_prev_value.view(bsz, 1, -1)


            h_0 = torch.cat([h_0_prev_key, h_0_prev_value], 1)

            #print(h_0.shape)
            h_0 = h_0.permute([1, 0, 2])

            #print(h_0.shape)

            output_list_new.append(h_0)



        

        






        '''
            input_ids_temp =  self.forzen_gpt2_tokenizer(input_token, return_tensors="pt")["input_ids"]

            #print(input_ids_temp.shape)

            forzen_decoder_input_ids = shift_tokens_right(input_ids_temp, self.pad_token_id)

            forzen_output = self.forzen_gpt2(input_ids_temp.to(self.forzen_gpt2.device), decoder_input_ids=forzen_decoder_input_ids.to(self.forzen_gpt2.device), use_cache=True,
            return_dict=True)

            forzen_output = forzen_output.past_key_values

            output_list_new = []


            #print(forzen_output)

            for item in forzen_output:
                h_0_prev_key  = item['encoder_decoder']['prev_key']
                #a.permute([])
                h_0_prev_value = item['encoder_decoder']['prev_value']
                #h_0_prev_key = h_0_prev_key[:, :, -1, :]


                #print(h_0_prev_key.shape)

                h_0_prev_key = torch.mean(h_0_prev_key, 2)
                #h_0_prev_value = h_0_prev_value[:, :, -1, :]

                h_0_prev_value = torch.mean(h_0_prev_value, 2)

                h_0_prev_key = h_0_prev_key.view(bsz, 1, -1)
                h_0_prev_value = h_0_prev_value.view(bsz, 1, -1)


                h_0 = torch.cat([h_0_prev_key, h_0_prev_value], 1)

                #print(h_0.shape)
                h_0 = h_0.permute([1, 0, 2])

                print(h_0.shape)

                output_list_new.append(h_0)
        '''

            #print(len(output_list_new))

            

        #print(top20.indices)



        #print("**************")

        #print(forzen_hiddensates.shape)

        #dst_ids = torch.zeros((bsz, 8)).cuda(self.device)
        #dst_ids = torch.zeros((bsz, 8), dtype=input_ids.dtype).cuda(self.device)
        #@print(input_ids)
        past_key_values_prompt, domain_value = self.get_prompt(bsz=bsz, h0_list=output_list_new, dst_ids=oracle_ids)





        

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, **kwargs)

        # output = gpt2_model(input_ids=input_ids,
        #                     past_key_values=past_key_values, attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids, position_ids=position_ids,
        #                    head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        #                    encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
        #                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        #                    return_dict=return_dict, **kwargs)

        return output, domain_value


    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1) # bsz, seqlen
        else:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"


        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output



class PrefixEmbTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False):
        super().__init__(config)

        print('under the PrefixEmbTuning model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0


        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512


        # if hasattr(config, 'mid_layers'):
        #     self.mid_layers = config.mid_layers
        # else:
        #     self.mid_layers = 1


        if False:
            if hasattr(config, '_my_arg_task_mode'):
                self.task_mode = config._my_arg_task_mode
            else:
                self.task_mode = 'under-specified'
                print('the task is underspecified')
                assert False

            if hasattr(config, 'train_weights'):
                self.train_weights = (config.train_weights == 'yes')
            else:
                self.train_weights = False
                assert False, 'train_weights should be specified.'

            print('train embedding is {}'.format(self.train_weights))

            if hasattr(config, '_my_arg_control'):
                print('control mode is on.')
                self.prefix_control = True
            else:
                self.prefix_control = False
                assert False, 'the control is underspecified'

        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts' or self.task_mode == 'summarization':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4


        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print('the is just for baseline checking!!! We reinitialize the LM embeddings and try cat '
                          'and peek.')
                    print('BASELINE'*100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)



            if self.mode_para == 1:
                print('mode_para=1, for dataless.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print('mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes...')

                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3

        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # DIFFERENT PARAMETRIZATION:
            if True:
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

            # DIFFERENT PARAMETRIZATION 2.
            elif True:
                print('UNDER PARAMETRIZATION 2')
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                input_word_lst = [['name', 'Type', 'price', 'customer rating', 'near', 'area', 'family friendly']]
                input_word_ids = tokenizer(input_word_lst, add_special_tokens=True, is_split_into_words=True, return_tensors='pt')['input_ids']
                self.input_embs = model_gpt2.transformer.wte(input_word_ids.to(model_gpt2.device))
                print(self.input_embs.shape)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p6_infix
                else:
                    self.get_prompt = self.get_prompt_p6



            # OLD CODE.
            # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
            # if self.use_infix:
            #     assert False, "just optimizing a set of parameter is not really related to infix position."
            #     self.get_prompt = self.get_prompt_p2_infix
            # else:
            #     self.get_prompt = self.get_prompt_p2

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


        ############################################################################



    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        '''
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        '''
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(bsz, -1, -1) #bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values

    def get_prompt_p2_infix(self, src_x, control_code=None, gpt2=None, bsz=None):
        '''
        Directly specifying/optimizing the input embeddings.
        :param control_code:
        :param gpt2:
        :param bsz:
        :return:
        '''
        assert bsz is not None
        temp_control = self.control_trans.unsqueeze(0).expand(bsz, -1, -1) #bsz, seqlen, emb
        temp_control = self.dropout(temp_control)
        src_embs = gpt2.wte(src_x)
        print(temp_control.shape, src_embs.shape)
        temp_control = torch.cat([src_embs, temp_control], dim=1)
        print(temp_control.shape)
        temp_result = gpt2(inputs_embeds=temp_control, use_cache=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        input_embs = self.control_trans(temp_control) #bsz, seqlen, emb_dim
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values


        return past_key_values

    def get_prompt_p3_infix(self, src_x, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb

            src_embs = gpt2.transformer.wte(src_x)
            input_embs = self.control_trans(temp_control) #bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            input_embs = torch.cat([src_embs, input_embs], dim=1)
            # print(input_embs.shape)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values


    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control) #bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values


    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            input_embs = self.control_trans(temp_control)  # bsz, seqlen, emb
            input_embs = self.dropout(input_embs)
            bsz, seqlen, _ = input_embs.shape
            # print(past_key_values.shape)
            temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
            past_key_values = temp_result.past_key_values
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # TODO-LISA
        self.format_mode = 'cat'
        if self.mode_para == 2:
            if self.format_mode == 'cat':
                past_key_values_prompt = self.get_prompt(src, cate_batch, gpt2=gpt2_model, bsz=bsz)
                attention_mask = torch.cat([src_attn, cate_attn, tgt_attn], dim=1)
            else:
                past_key_values_prompt = self.get_prompt(src, src, gpt2=gpt2_model, bsz=bsz)
                attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)
        else:

            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            bsz, seqlen = src.shape
            temp_attn = torch.ones(bsz, self.preseqlen).bool()
            attention_mask = torch.cat([src_attn, temp_attn, tgt_attn], dim=1)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output

    def forward(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output







