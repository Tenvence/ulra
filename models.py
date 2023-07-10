import torch
import torch.nn as nn
import transformers


class Model(nn.Module):
    def __init__(self, num_features, weight_init):
        super(Model, self).__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.score_fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.weight_memory = nn.Parameter(torch.full((num_features,), fill_value=torch.logit(torch.tensor(weight_init))), requires_grad=True)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask).last_hidden_state  # [B, len_seq, dim]
        x = bert_output.mean(dim=1)  # [B, dim]
        pred_score = self.score_fc(x)

        return pred_score, self.weight_memory.sigmoid()
