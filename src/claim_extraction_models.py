from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
import re

from utils import print_list, DEBUG_MODE, DEVICE

nltk.download('punkt')

def create_claim_extraction_prompt_template():

    proposition_examples = [
        {"document": 
            "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.", 
        "propositions": 
            "['Neil Armstrong was an astronaut.', 'Neil Armstrong walked on the Moon in 1969.', 'Neil Armstrong was the first person to walk on the Moon.', 'Neil Armstrong walked on the Moon during the Apollo 11 mission.', 'The Apollo 11 mission occurred in 1969.']"
        },
    ]

    example_proposition_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{document}"),
            ("ai", "{propositions}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_proposition_prompt,
        examples = proposition_examples,
    )

    system = """Please break down the following text into simple, self-contained propositions. Ensure that each proposition meets the following criteria:

        1. Express a Single Fact: Each proposition should state one specific fact or claim.
        2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
        3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
        4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
        5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            few_shot_prompt,
            ("human", "{document}"),
        ]
    )

    return prompt


class Claims(BaseModel):
    """List of all the claims/propositions made in the given document."""
    claims: list[str] = Field(
        description="List of claims/propositions (factual, self-contained, and concise information)"
    )

class ClaimExtractor(ABC):
    @abstractmethod
    def extract_claims(self, text: str) -> list[str]:
        pass

class OllamaClaimExtractor(ClaimExtractor):
    def __init__(self, model_name: str) -> None:

        structured_llm = OllamaLLM(model=model_name, temperature=0.0).with_structured_output(Claims)
        parser = PydanticOutputParser(pydantic_object=Claims)
        prompt = create_claim_extraction_prompt_template()
        self.proposition_generator = prompt | structured_llm | parser

    def extract_claims(self, text: str) -> list[str]:
        if text.strip() == "":
            if DEBUG_MODE:
                print("=> No text provided for claim extraction. Returning empty list.")
            return []

        claims: list[str] = self.proposition_generator.invoke({
            "document": text
        }).claims

        if DEBUG_MODE:
            print(f"\n=> Extracted claims from the text:\n")
            print_list(claims)

        # No longer cleaning up claims because it could mess up the decontextualisation process
        # # Cleaning up claims
        # dont_start_with = ['-', '- ', 'The ', 'speaker ', 'text ', 'post ', 'article ', 'blog post ', 'blog article ', 'blog post article ']
        # for substring in dont_start_with:
        #     if all(claim.startswith(substring) for claim in claims):
        #         claims = [claim[len(substring):] for claim in claims]

        # if all(claim.endswith(".") for claim in claims):
        #     claims = [claim[:-1] for claim in claims]

        # claims = [claim.strip() for claim in claims if claim.strip()]

        # if DEBUG_MODE:
        #     print(f"\n=> Cleaned claims from the text:\n")
        #     print_list(claims)

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



class Decontextualiser(ABC):
    @abstractmethod
    def decontextualise(self, before: str, text: str, after: str) -> str:
        pass

class NonDecontextualiser(Decontextualiser):
    def decontextualise(self, before: str, text: str, after: str) -> str:
        if DEBUG_MODE:
            print("=> NonDecontextualiser called. Returning original text without changes.")
        return text



class FalsifiabilityChecker(ABC):
    @abstractmethod
    def is_falsifiable(self, claim: str) -> bool:
        pass
    
class NonFalsifiabilityChecker(FalsifiabilityChecker):
    def is_falsifiable(self, claim: str) -> bool:
        if DEBUG_MODE:
            print("=> NonFalsifiabilityChecker called. Returning True for all claims.")
        return True