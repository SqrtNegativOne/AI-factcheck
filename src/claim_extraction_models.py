from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
import re

from utils import print_list, DEBUG_MODE, CLAIM_EXTRACTION_TEMPLATE_PATH, DEVICE

nltk.download('punkt')

class Claims(BaseModel):
    claims: list[str] = Field(description="List of claims extracted from the text.")

class ClaimExtractor(ABC):
    @abstractmethod
    def extract_claims(self, text: str) -> list[str]:
        pass

class Decontextualiser(ABC):
    @abstractmethod
    def decontextualise(self, text: str) -> str:
        pass

class OllamaDecontextualiser(Decontextualiser):
    def __init__(self, model_name: str) -> None:
        with open(CLAIM_EXTRACTION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            CLAIM_EXTRACTION_TEMPLATE: str = f.read()
        prompt = PromptTemplate.from_template(CLAIM_EXTRACTION_TEMPLATE) # TODO: fix ⚠️

        llm = OllamaLLM(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=Claims)
        self.chain = prompt | llm | self.parser

    def decontextualise(self, text: str) -> str:
        if text.strip() == "":
            if DEBUG_MODE:
                print("=> No text provided for decontextualisation. Returning empty string.")
            return ""

        claims: list[str] = self.chain.invoke({"text": text, "format_instructions": self.parser.get_format_instructions()}).claims
        return " ".join(claims)

class OllamaClaimExtractor(ClaimExtractor):
    def __init__(self, model_name: str) -> None:
        with open(CLAIM_EXTRACTION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            CLAIM_EXTRACTION_TEMPLATE: str = f.read()
        prompt = PromptTemplate.from_template(CLAIM_EXTRACTION_TEMPLATE)

        llm = OllamaLLM(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=Claims)
        self.chain = prompt | llm | self.parser

    def extract_claims(self, text: str) -> list[str]:

        if text.strip() == "":
            if DEBUG_MODE:
                print("=> No text provided for claim extraction. Returning empty list.")
            return []

        claims: list[str] = self.chain.invoke({"text": text, "format_instructions": self.parser.get_format_instructions()}).claims

        # Cleaning up claims
        dont_start_with = ['-', '- ', 'The ', 'speaker ', 'text ', 'post ', 'article ', 'blog post ', 'blog article ', 'blog post article ']
        for substring in dont_start_with:
            if all(claim.startswith(substring) for claim in claims):
                claims = [claim[len(substring):] for claim in claims]
        
        if all(claim.endswith(".") for claim in claims):
            claims = [claim[:-1] for claim in claims]
        
        claims = [claim.strip() for claim in claims if claim.strip()]

        if DEBUG_MODE:
            print(f"\n=> Cleaned claims from the text:\n")
            print_list(claims)
        
        return claims

class Gemma_7B_APS_Claim_Extractor(ClaimExtractor):
    def __init__(self):
        model_id = 'google/gemma-7b-aps-it'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16, #torch.bfloat16 isn't supported on T4?
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        self.start_marker = '<s>'
        self.end_marker = '</s>'
        self.separator = '\n'

    def create_propositions_input(self, text: str) -> str:
        input_sents = nltk.tokenize.sent_tokenize(text)
        propositions_input = ''
        for sent in input_sents:
            propositions_input += f'{self.start_marker} ' + sent + f' {self.end_marker}{self.separator}'
        propositions_input = propositions_input.strip(f'{self.separator}')
        return propositions_input

    def process_propositions_output(self, text):
        pattern = re.compile(f'{re.escape(self.start_marker)}(.*?){re.escape(self.end_marker)}', re.DOTALL)
        output_grouped_strs = re.findall(pattern, text)
        predicted_grouped_propositions = []
        for grouped_str in output_grouped_strs:
            grouped_str = grouped_str.strip(self.separator)
            props = [x[2:] for x in grouped_str.split(self.separator)]
            predicted_grouped_propositions.append(props)
        return predicted_grouped_propositions

    def extract_claims(self, text: str) -> list[str]:
        messages = [{'role': 'user', 'content': self.create_propositions_input(text)}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True, return_dict=True).to(DEVICE)

        output = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_text = self.tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        result = self.process_propositions_output(generated_text)

        flattened_result: list[str] = [claim for sublist in result for claim in sublist]
        return flattened_result

