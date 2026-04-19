from datasets import load_dataset
import json
import os

def download_and_clean():

    """
    Downloads the MedQuAD dataset from HuggingFace and cleans it.
    We only keep rows that have both a question AND an answer.
    We save the result to data/processed/medquad_clean.json
    """

    print("Downloading MedQuAD dataset from HuggingFace...")

    #loading keivalya/MedQuad-MedicalQnADataset dataset from HF

    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

    print(f"Raw dataset size: {len(dataset)} rows")


    cleaned = []

    for row in dataset:

        #getting the Question and Answer from each row in the dataset
        question = row.get("Question","").strip()
        answer = row.get("Answer","").strip()

        #only keeping rows where both question and ans are present 
        #and length of ans is atleast 20 characters to avoid vague one-word answers
        if question and answer and len(answer)>20:
            cleaned.append(
                {
                    "question":question,
                    "answer" : answer
                }
            ) 

    print(f"Cleaned Dataset: {len(cleaned)}")   

    os.makedirs("data/processed", exist_ok=True)

    #path to save clean data
    output_path = "data/processed/medquad_clean.json"

    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2) # indecnt=2 for pretty-printing

    print(f"Saved cleaned data to {output_path}")

    return output_path

if __name__ == "__main__":
    download_and_clean()    


       