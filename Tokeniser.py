import re

with open ("the-verdict.txt.rtf", "r", encoding="utf-8") as f:
    raw_text = f.read()
print ("Total number of characters:", len(raw_text))
print (raw_text[:99])

#Simple tokenisation
text = "Hello world. Check out this test."
result = re.split(r'(\s)', text)
print(result)