import re


class SimpleTokeniserV1:
    def __init__(self, vocab):
        # Stores the vocab as a class attribute for access in the encode and decode methods
        self.str_to_int = vocab
        # Creates an inverse vocabulary that maps token IDs back to tokens
        self.int_to_str = {i: s for s,i in vocab.items()}

    # Processes input text into token ids
    def encode(self, text):
        # Updated regex for correct tokenization
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Removes spaces before specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])',r'\1', text)
        return text


tokeniser = SimpleTokeniserV1(vocab)
text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokeniser.encode(text)
print(ids)
print(tokeniser.decode(ids))

text = "Hello, would you like a cuppa?"
print(tokeniser.encode(text))