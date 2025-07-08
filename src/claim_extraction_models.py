from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_claim_extraction_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

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
        from langchain_ollama import OllamaLLM
        from langchain.output_parsers import PydanticOutputParser

        llm = OllamaLLM(model=model_name, temperature=0) # .with_structured_output(Claims) # OllamaLLM does not support structured output yet
        parser = PydanticOutputParser(pydantic_object=Claims)
        prompt = create_claim_extraction_prompt_template()
        self.proposition_generator = prompt | llm | parser

    def extract_claims(self, text: str) -> list[str]:
        if text.strip() == "":
            logger.warning("=> No text provided for claim extraction. Returning empty list.")
            return []

        try:
            claims: list[str] = self.proposition_generator.invoke({
                "document": text
            }).claims
        except Exception as e:
            logger.error(f"Error during claim extraction: {e}")
            return []

        logger.info(f"\n=> Extracted claims from the text:\n")
        logger.info(f"Extracted claims from the text:\n{claims}")

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
        #     logger.info(f"\n=> Cleaned claims from the text:\n")
        #     logger.info(f"Cleaned claims from the text:\n{claims}")

        return claims

class Gemma_APS_Claim_Extractor(ClaimExtractor):
    def __init__(self, use_7b_params = False) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import nltk
        self.nltk = nltk
        import re
        self.re = re

        self.nltk.download('punkt')
        model_id = 'google/gemma-2b-aps-it' if not use_7b_params else 'google/gemma-7b-aps-it'
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def create_propositions_input(self, text: str) -> str:
        input_sents = self.nltk.tokenize.sent_tokenize(text)
        propositions_input = ''
        for sent in input_sents:
            propositions_input += f'{self.start_marker} ' + sent + f' {self.end_marker}{self.separator}'
        propositions_input = propositions_input.strip(f'{self.separator}')
        return propositions_input

    def process_propositions_output(self, text):
        pattern = self.re.compile(f'{self.re.escape(self.start_marker)}(.*?){self.re.escape(self.end_marker)}', self.re.DOTALL)
        output_grouped_strs = self.re.findall(pattern, text)
        predicted_grouped_propositions = []
        for grouped_str in output_grouped_strs:
            grouped_str = grouped_str.strip(self.separator)
            props = [x[2:] for x in grouped_str.split(self.separator)]
            predicted_grouped_propositions.append(props)
        return predicted_grouped_propositions

    def extract_claims(self, text: str) -> list[str]:
        messages = [{'role': 'user', 'content': self.create_propositions_input(text)}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True, return_dict=True).to(self.device)

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
        logging.debug("=> NonDecontextualiser called. Returning original text without changes.")
        return text



class FalsifiabilityChecker(ABC):
    @abstractmethod
    def is_falsifiable(self, claim: str) -> bool:
        pass
    
class NonFalsifiabilityChecker(FalsifiabilityChecker):
    def is_falsifiable(self, claim: str) -> bool:
        logging.debug("=> NonFalsifiabilityChecker called. Returning True for all claims.")
        return True
    

if __name__ == "__main__":
    logging.error("This module is not intended to be run directly. Please import it in your main application.")