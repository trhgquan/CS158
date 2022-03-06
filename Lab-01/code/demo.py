import kenlm
import math

model = kenlm.Model('model.arpa')

# sentence = "\"I can't understand why my grandmother never gambles.\""
sentence = "my grandmother never gambles ."

# true_score = math.pow(10.0, model.score(sentence))

# print("Model score:", true_score)
print("Model score:", model.score(sentence))

perplexity = model.perplexity(sentence)

print("Sentence perplexity:", perplexity)
