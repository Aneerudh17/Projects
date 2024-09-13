from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)


conversation =[]  


while True:
    history_string = "\n".join(conversation)

    input_string = input("> ")

    if input_string.lower() == "goodbye":
        print("Goodbye! Have a great day!")
        break

    conversation.append(input_string)


    inputs = tokenizer.encode_plus(history_string,input_string, return_tensors="pt")

    outputs = model.generate(**inputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)


    conversation.append(response)
