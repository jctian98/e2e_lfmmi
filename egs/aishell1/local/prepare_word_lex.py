import sys

"""
Make a word-level lexicon for MMI training. 
Previous lexicon accepts phones, here this lexicon accepts words.
"""

in_f = sys.argv[1]
out_f = sys.argv[2]
char_out_f = sys.argv[3]

cnt = 0
writer = open(out_f, 'w', encoding="utf8")
char_writer = open(char_out_f, 'w', encoding="utf8")
for line in open(in_f, encoding="utf8"):
    cnt += 1
  
    # The first two lines should be kept: special tokens   
    if cnt <= 2:
        writer.write(line)
        char_writer.write(line)
        continue

    word = line.split()[0]
    line = word + " " + " ".join(list(word)) + "\n"
    writer.write(line)

    if len(word) == 1:
        line = f"{word} {word}\n"
        char_writer.write(line)

writer.close()
char_writer.close()
 
