from typing import Optional

import torch
import torch.nn as nn
from transformers.models.siglip.modeling_siglip import SiglipModel, SiglipOutput


class LocalSiglipModel(SiglipModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        labels: Optional[torch.LongTensor] = None,  # for custom loss
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_text_to_text = torch.matmul(text_embeds, text_embeds.t().to(text_embeds.device))
        logits_text_to_text = logits_text_to_text * logit_scale.exp().to(text_embeds.device) + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        # if return_loss and labels is not None:
        if labels is not None:
            temp_loss = custome_loss(logits_per_text, labels)
            diversity_loss = contrastive_loss(logits_text_to_text) / 3.0
            loss = (temp_loss + diversity_loss) / 2.0

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return SiglipOutput(
            loss=loss,  # type: ignore
            logits_per_image=logits_per_image,  # type: ignore
            logits_per_text=logits_per_text,  # type: ignore
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

def custome_loss(
    logits_per_text: torch.Tensor,
    labels: torch.Tensor,
):
    loglik = torch.nn.functional.logsigmoid(logits_per_text.t() * labels)
    nll = -torch.sum(loglik, dim=-1)
    loss = nll.mean()
    return loss

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

