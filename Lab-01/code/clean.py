import string, re

filename_raw = 'qos-raw.txt'
filename_clean = 'queenofspades.txt'

data = []

with open(filename_raw, 'r+') as f:
    for line in f.readlines():
        line = line.lower()
        line = re.sub('([.,!?()])', r' \1 ', line)
        line = re.sub('\s{2,}', ' ', line)
        line = line.strip() 
        data.append(line)

with open(filename_clean, 'w+') as f:
    for line in data:
        print(line, file = f)
