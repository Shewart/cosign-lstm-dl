import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

class LLM:
    def __init__(self):
        # Initialize the ChatOpenAI model using your API key
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" or "gpt-4o" depending on your access
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # Define a prompt template that instructs the model to convert English to ASL Gloss
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert in ASL grammar. Convert the following sentence into ASL Gloss. "
                "Keep the meaning, but rephrase it following ASL word order (object, then subject, then verb). "
                "Remove unnecessary words like 'is' and 'are'. Replace 'I' with 'ME' and do not include articles."
            ),
            ("human", "{transcription}")
        ])
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
