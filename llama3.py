import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class Llama3_pipline():
    '''
    A class of create llama3 pipeline
    '''

    def __init__(self, model_name, hf_token):
        '''
        initial vectordb class

        Args:
        model_name (str): llm model from huggingface
        hf_token (dict): huggingface token
        '''
        self.model_name = model_name
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        assert self.hf_token is not None

        self._create_qmodel_tokenizer()
        self._create_pipeline()

    def _create_qmodel_tokenizer(self):

        compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and True:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16")
                print("=" * 80)

        device_map = {"": 0}

        # Load LLaMA base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            use_auth_token=self.hf_token
        )
        self.model.config.use_cache = False

        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_auth_token=self.hf_token
        )

    def _create_pipeline(self):
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=200
        )
