import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
 
def sent_scoring(model, tokenizer, text, cuda):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob
 

if __name__ == '__main__':
    # Initialisiere Modell
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
    model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model.to('cuda')

    # Berechne Score
    print(sent_scoring(model, tokenizer, "Goldfische sind pflegeleicht.", cuda))
    print(sent_scoring(model, tokenizer, "Gold Fische sind pflegeleicht.", cuda))
	
	
