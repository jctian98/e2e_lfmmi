import sys

in_f = sys.argv[1]
writers = []
for f in sys.argv[2:]:
    writer = open(f, 'w', encoding='utf-8')
    writers.append(writer)
num_writers = len(writers)

for i, line in enumerate(open(in_f, encoding='utf-8')):
    writer = writers[i % num_writers]
    writer.write(line)

for w in writers:
    writer.close()
    
