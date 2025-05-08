
import torch
# TODO - SHOULD BE FURTHER IMPROVED.
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>", "<formalization>", "</formalization>", "<answer>", "</answer>"),
                 max_len=8000, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        
        self.sptids_dict['<｜begin▁of▁sentence｜>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<｜end▁of▁sentence｜>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        self.sptids_dict['<｜User｜>'] = torch.tensor([self.text_tokenizer.convert_tokens_to_ids('<｜User｜>')])
        self.sptids_dict['<｜Assistant｜>'] = torch.tensor([self.text_tokenizer.convert_tokens_to_ids('<｜Assistant｜>')])
        self.max_len = max_len
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id
        self.assistant_id = self.text_tokenizer.convert_tokens_to_ids('<｜Assistant｜>')
        system_message = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        self.system_ids = self.text_tokenizer(system_message, add_special_tokens=False)['input_ids']
    
    def t2i_prompt(self, text_ids, image_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User] [task token][text tokens] [Assistant] [soi] [image tokens] [eoi] <｜end▁of▁sentence｜>
            temp_text_ids =  [self.text_tokenizer.bos_token_id] + self.system_ids + [int(self.sptids_dict['<｜User｜>'])] + [int(self.sptids_dict['<|t2i|>'])] + \
                text_ids[i] 
            temp_ids  = torch.cat([
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<｜Assistant｜>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                # torch.tensor(temp_text_ids).to(device),
                torch.ones_like(torch.tensor(temp_text_ids)).to(device) * self.ignore_id,
                self.sptids_dict['<｜Assistant｜>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)  
        
            assert temp_ids.shape[0] == temp_label_ids.shape[0]
            
            current_len = temp_ids.shape[0]       
            if self.max_len >= current_len:
                temp_masks =  [0] * (self.max_len - current_len) + [1] * current_len 
                temp_ids = torch.cat([torch.tensor([self.pad_id] * (self.max_len - current_len)).to(device),
                                      temp_ids                 
                    ], dim=0)
                temp_label_ids = torch.cat([torch.tensor([self.ignore_id] * (self.max_len - current_len)).to(device),
                                            temp_label_ids])
                
            else:
                # should add the eos token
                temp_ids = torch.cat([temp_ids[:self.max_len - 1],
                                      self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)],
                                     dim=0)
                temp_label_ids = torch.cat([temp_label_ids[:self.max_len-1],
                                            self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)])
                temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long), torch.cat(label_ids, dim=0).to(torch.long)
        
    
    def t2i_gen_prompt(self, text_ids):
        sequence_ids = []
        attention_masks = []
        assert len(text_ids) == 1
        for i in range(len(text_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User] [task token] [text tokens] [Assistant] [soi]
            temp_text_ids =  [self.text_tokenizer.bos_token_id] + self.system_ids + [int(self.sptids_dict['<｜User｜>'])] + [int(self.sptids_dict['<|t2i|>'])] + \
                 text_ids[i]
            temp_ids = torch.cat([
                torch.tensor(temp_text_ids),
                self.sptids_dict['<｜Assistant｜>'],
                self.sptids_dict['<|soi|>'],
            ], dim=0)
            temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long)

    def mmu_prompt(self, image_ids, instruction_ids, response_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        assert image_ids.shape[0] == len(instruction_ids) == len(response_ids)
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User] [task token] [soi] [image tokens] [eoi][text tokens] <｜end▁of▁sentence｜> 
            response_ids[i] = [self.assistant_id] + response_ids[i]
            temp_text_ids = instruction_ids[i] + response_ids[i]
            temp_label_ids =  [self.ignore_id] * len(instruction_ids[i]) + response_ids[i]
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'].to(device),
                torch.tensor(self.system_ids).to(device),
                self.sptids_dict['<｜User｜>'].to(device),
                self.sptids_dict['<|mmu|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(torch.tensor(self.system_ids)).to(device) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_label_ids).to(device),
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)

            assert temp_ids.shape[0] == temp_label_ids.shape[0]
            
            current_len = temp_ids.shape[0]     
            
            if self.max_len >= current_len:
                temp_masks =  [0] * (self.max_len - current_len) + [1] * current_len 
                temp_ids = torch.cat([torch.tensor([self.pad_id] * (self.max_len - current_len)).to(device),
                                      temp_ids], dim=0)
                temp_label_ids = torch.cat([torch.tensor([self.ignore_id] * (self.max_len - current_len)).to(device),
                                            temp_label_ids])
                
            else:
                # should add the eos token
                temp_ids = torch.cat([temp_ids[:self.max_len - 1],
                                      self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)],
                                     dim=0)
                temp_label_ids = torch.cat([temp_label_ids[:self.max_len-1],
                                            self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)])
                temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long), torch.cat(label_ids, dim=0).to(torch.long)
    
    def mmu_gen_prompt(self, image_ids, instruction_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        assert len(instruction_ids) == 1
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User][task token] [soi] [image tokens] [eoi] [text tokens] [Assistant]
            temp_text_ids =  instruction_ids[i]
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'].to(device),
                torch.tensor(self.system_ids).to(device),
                self.sptids_dict['<｜User｜>'].to(device),
                self.sptids_dict['<|mmu|>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_text_ids).to(device),
                self.sptids_dict['<｜Assistant｜>'].to(device),
            ], dim=0)
            temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long)
        
        
        
        
           
    def mix_prompt(self, image_ids, instruction_ids, response_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        assert image_ids.shape[0] == len(instruction_ids) == len(response_ids)
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User] [task token] [instrution tokens] [soi] [image tokens] [eoi] [response tokens] <｜end▁of▁sentence｜> 
            instruction = torch.tensor(instruction_ids[i]).to(device)
            response = torch.tensor(response_ids[i]).to(device)
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'].to(device),
                torch.tensor(self.system_ids).to(device),
                self.sptids_dict['<｜User｜>'].to(device),
                self.sptids_dict['<|mix|>'].to(device),
                instruction,
                self.sptids_dict['<｜Assistant｜>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                response,
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(torch.tensor(self.system_ids)).to(device) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(instruction).to(device) * self.ignore_id,
                self.sptids_dict['<｜Assistant｜>'].to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                response,
                self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)
            ], dim=0)
            
            assert temp_ids.shape[0] == temp_label_ids.shape[0]
            
            current_len = temp_ids.shape[0]    
            
            if self.max_len >= current_len:
                temp_masks =  [0] * (self.max_len - current_len) + [1] * current_len 
                temp_ids = torch.cat([torch.tensor([self.pad_id] * (self.max_len - current_len)).to(device),
                                      temp_ids], dim=0)
                temp_label_ids = torch.cat([torch.tensor([self.ignore_id] * (self.max_len - current_len)).to(device),
                                            temp_label_ids])
                
            else:
                # should add the eos token
                temp_ids = torch.cat([temp_ids[:self.max_len - 1],
                                      self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)],
                                     dim=0)
                temp_label_ids = torch.cat([temp_label_ids[:self.max_len-1],
                                            self.sptids_dict['<｜end▁of▁sentence｜>'].to(device)])
                temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0)) 
            
        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long), torch.cat(label_ids, dim=0).to(torch.long)
    
    def mix_gen_prompt(self, instruction_ids):
        sequence_ids = []
        attention_masks = []
        assert len(instruction_ids) == 1
        for i in range(len(instruction_ids)):
            # prompting -- <｜begin▁of▁sentence｜> [system tokens] [User] [task token] [instrution tokens] [Assistant]
            temp_text_ids =  instruction_ids[i]
            temp_ids = torch.cat([
                self.sptids_dict['<｜begin▁of▁sentence｜>'],
                torch.tensor(self.system_ids),
                self.sptids_dict['<｜User｜>'],
                self.sptids_dict['<|mix|>'],
                torch.tensor(temp_text_ids),
                self.sptids_dict['<｜Assistant｜>'],
            ], dim=0)
            temp_masks = [1] * temp_ids.shape[0]
            temp_masks = torch.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
        return torch.cat(sequence_ids, dim=0).to(torch.long), torch.cat(attention_masks, dim=0).to(torch.long)

    def __call__(self, input, task):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0], add_special_tokens=False)['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids)
            

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer([input], add_special_tokens=False)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids)
        
        elif task == "mmu":
            image_ids = input[0]
            instruction_ids = self.text_tokenizer(input[1], add_special_tokens=False)['input_ids']
            response_ids = self.text_tokenizer(input[2], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_prompt(image_ids, instruction_ids, response_ids)
            
        elif task == 'mmu_gen':
            image_ids = input[0]
            instruction_ids = self.text_tokenizer([input[1]], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mmu_gen_prompt(image_ids, instruction_ids)
        elif task == 'mix':
            image_ids = input[0]
            instruction_ids = self.text_tokenizer(input[1], add_special_tokens=False)['input_ids']
            response_ids = self.text_tokenizer(input[2], add_special_tokens=False)['input_ids']
            sequence_ids_with_masks = self.mix_prompt(image_ids, instruction_ids, response_ids)
        elif task == 'mix_gen':
            text_ids = self.text_tokenizer([input], add_special_tokens=False)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.mix_gen_prompt(text_ids)
            
            
        else:
            raise NotImplementedError

        return sequence_ids_with_masks

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    uni_prompting = UniversalPrompting(tokenizer, max_len=20,
                                       ignore_id=-100)
    
    instructions = ['一只小猪',
                  '生成随机']
    responses = ['小黑猫',
                 '胖胖的']
    image_input = torch.randint(0, 10, (2, 10))
    input, attention_mask = uni_prompting(instructions[0], task='t2i_gen')
    print('*'*10 + ' t2i_gen ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    
    input, attention_mask, label = uni_prompting((instructions, image_input, image_input), task='t2i')
    print('*'*10 + ' t2i ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    
    
    input, attention_mask, label = uni_prompting((image_input, instructions, responses), task='mmu')
    print('*'*10 + ' mmu ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    
    input, attention_mask = uni_prompting([image_input[0, :].unsqueeze(0), instructions[0]], task='mmu_gen')
    print('*'*10 + ' mmu_gen ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    
    input, attention_mask, label = uni_prompting((image_input, instructions, responses), task='mix')
    print('*'*10 + ' mix ' + '*'*10)
    print(f'input: {input}')
    print(f'input shape: {input.shape}')
    print(f'attention_mask: {attention_mask}')
    print(f'attention_mask shape: {attention_mask.shape}')
    print(f'label: {label}')
    print(f'label shape: {label.shape}')
    
    