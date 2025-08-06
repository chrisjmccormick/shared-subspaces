"""#### *ForMaskedLM"""

# Import the return object, MaskedLMOutput
from typing import Optional
from dataclasses import dataclass

from transformers.modeling_outputs import (
    #BaseModelOutputWithPastAndCrossAttentions,
    #BaseModelOutputWithPoolingAndCrossAttentions,
    #CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    #MultipleChoiceModelOutput,
    #NextSentencePredictorOutput,
    #QuestionAnsweringModelOutput,
    #SequenceClassifierOutput,
    #TokenClassifierOutput,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shared_space_config import SharedSpaceEncoderConfig
from models.shared_space_encoder import (
    SharedSpaceEncoderPreTrainedModel,
    SharedSpaceEncoderModel
)


class SharedSpaceEncoderForMaskedLM(SharedSpaceEncoderPreTrainedModel):
    """
    The `*MaskedLM` object:
        - Initializes:
            - A `*Model` object from the given config.
            - (It doesn't create a new LM head--we just use the vocabulary)
    """

    def __init__(self, config: SharedSpaceEncoderConfig) -> None:

        # Call the `*PreTrainedModel` init.
        super().__init__(config)

        # Create the `*Model`. Everything we need is already there.
        self.encoder_model = SharedSpaceEncoderModel(config)

        # Call the `*PreTrainedModel` init
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        The `labels` are token ids with the `[MASK]` token id at the masked
        positions, and -100 (we tell the loss function this) everywhere else.
        The `attention_mask`...

        Inputs:
               input_ids: [batch_size, seq_len]
          attention_mask: [batch_size, 1, 1, seq_len]
                 labels : [batch_size, seq_len]

        The `logits` are the prediction scores over the vocabulary.
        The `loss` is a scalar value... per sample in the batch? It has had
        the mask applied to it? cross-entropy. Only when labels provided, otherwise it's...

        Outputs:
           logits: [batch_size, seq_len, vocab_size]  Predction scores over vocab
             loss: [batch_size]? Or average over batch?
        """

        # Run the input through the whole model.
        hidden_states = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs, # TODO - What can be passed here?
        )

        # Retrieve the vocabulary.
        W_E = self.encoder_model.vocab_embed.weight

        # The hidden states are model size. If the vocabulary was decomposed,
        # We need to down project, and then multiply with the vocabulary latents.
        # Otherwise, multiply directly with the vocabulary embeddings.
        # --- Shared projection → logits  -------------------

        if self.encoder_model.vocab_proj is not None:
            #  B - batch_size
            #  T - sequence length
            #  D - model_size
            #  C - latent_size
            #  V - vocab_size

            # Linear stores the transpose of its projection, so vocab_proj
            # is functionally [C x D], but stored as [D x C]
            # So the vocabulary latent space projection, W_E_proj, is [D x C]
            W_E_proj = self.encoder_model.vocab_proj.weight

            # Project the tokens output by the model into the vocabulary
            # subspace.
            #
            # Inputs:
            #    hidden_states   [B, T, D]
            #         W_E_proj         [D, C]
            # Outputs:
            #        h_latents   [B, T, C]
            #
            #  TODO - Fuse with the next op if beneficial.
            #h_latents = einsum('btd,dc->btc', hidden_states, W_E_proj)

            # Multiply each token latent with every vocabulary latent to
            # get the per-token logit scores over the vocabulary.
            #
            # Inputs:
            #      h_latents   [B, T, C]
            #            W_E   [V, C]
            # Outputs:
            #    logits  [B, T, V]
            #logits = einsum('btc,vc->btv', h_latents, self.vocab_embed.weight)

            logits = torch.einsum('btd,dc,vc->btv', hidden_states, W_E_proj, W_E)

        # If there's no vocabulary subspace,
        else:
            # Multiply the hidden states with the vocabulary.
            #
            # Inputs:
            #    hidden_states   [B, T, D]
            #              W_E   [V, D]
            # Outputs:
            #    logits  [B, T, V]
            logits = torch.einsum('btd,vd->btv', hidden_states, W_E)

        vocab_size = W_E.size(0)

        # If labels are provided,
        loss = None
        if labels is not None:
            # Flatten everything for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



        # Return the output as a dictionary.
        output = {
            "logits": logits,
            "loss": loss,
        }
        return output

"""#### `*ForSequenceClassification`

Copied from BERT
"""

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput

# Import Union
from typing import Union

class SharedSpaceEncoderForSequenceClassification(SharedSpaceEncoderPreTrainedModel):
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Call the `*PreTrainedModel` init.
        super().__init__(config)

        # Create the `*Model`. Everything we need is already there.
        self.encoder_model = SharedSpaceEncoderModel(config)

        # Incorporate BERT's pooling layer.
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        #position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        #inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # ============================
        #        Evaluate
        # ===========================
        hidden_states = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        # ===============================
        #      Non-Linearity ("Pooler")
        # ==============================
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # ====================
        #      Classifier
        # ====================
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            # TODO
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )



