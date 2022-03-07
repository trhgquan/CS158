import kenlm
import math

model = kenlm.Model('model.arpa')

# sentence = "\"I can't understand why my grandmother never gambles.\""
sentence = "my grandmother never gambles ."

score = model.score(sentence)

print("Sentence score:", score)

print("Sentence true score:", 10**score)

perplexity = model.perplexity(sentence)

print("Sentence perplexity:", perplexity)
