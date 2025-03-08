import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

class LLM:
    def __init__(self):
        # Initialize the ChatOpenAI model using your API key
        self.llm = ChatOpenAI(
            model="gpt-4o",  # You can change this to gpt-3.5-turbo or another model if desired
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # Define a prompt that instructs the model to convert text to ASL Gloss
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are meant to convert text from English to ASL Gloss grammar. Do not change meaning or move periods. I will send you a phrase, please rephrase it to follow ASL grammar order: object, then subject, then verb. Remove words like IS and ARE that are not present in ASL. Replace I with ME. Do not add classifiers. Everything should be in plain English. Output nothing but the rephrased phrase."
            ),
            ("human", "{transcription}")
        ])
        # Chain together the prompt and the language model
        self.chain = self.prompt | self.llm

    def gloss(self, transcription):
        """
        Convert a list of English words (transcription) into ASL Gloss tokens.
        """
        raw_transcription = " ".join(transcription)
        response = self.chain.invoke({"transcription": raw_transcription})
        output = response.content.strip()
        print("ASL Gloss:", output)
        return output.split()

if __name__ == "__main__":
    llm_instance = LLM()
    # Example: Convert a sample English phrase into ASL Gloss
    sample_transcription = ["I", "am", "going", "to", "the", "store."]
    asl_gloss = llm_instance.gloss(sample_transcription)
    print("Resulting ASL Gloss Tokens:", asl_gloss)
