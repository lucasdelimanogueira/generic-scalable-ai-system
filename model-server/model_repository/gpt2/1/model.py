import numpy as np
import itertools
import torch
import triton_python_backend_utils as pb_utils
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

    def execute(self, request):
        request = request[0]
        inputs_text = pb_utils.get_input_tensor_by_name(request, "input_text")
        inputs_text = inputs_text.as_numpy().astype(str)
        inputs_text = list(itertools.chain(*inputs_text))
        
        input_ids = self.tokenizer(
            inputs_text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**input_ids)
        outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        inference_response = pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor(
                "output_text",
                np.array([s.encode('utf-8') for s in outputs_text])
            )
        ])
        response = [inference_response]

        return response
