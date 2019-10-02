import os
import wave

from os import listdir
from os.path import isfile, join

'''
path = "/Users/erdemdemir/Desktop/bird-sounds"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in d:
        files.append(os.path.join(r, file))

files.sort()

#for f in files:
#    print(f)
'''




'''
file_to_write = open("/Users/erdemdemir/Desktop/bird-dir/bird-types.txt","w+")

for i in files:
    file_to_write.write("{}\n".format(i))

file_to_write.close()
'''

file_to_read = open("/Users/erdemdemir/Desktop/bird-dir/bird-types.txt","r")

file_n = []

for t in file_to_read:
    file_n.append(t[38:len(t)-1])

'''
file_n.sort()
for t in file_n:
    print(t)
'''


'''
#Created type .txt s
for k in file_n:
    x = "/Users/erdemdemir/Desktop/bird-dir/" + k + ".txt"
    print(x)
    file_to_wr = open(x,"w+")
    file_to_wr.close()
'''



'''
path1 = "/Users/erdemdemir/Desktop/bird-sounds/" + file_n[14]

files1 = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path1):
    for file in d:
        print()
        #files1.append(os.path.join(r, file))

files1.sort()

for f in files1:
    print(f)

onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]

onlyfiles.sort()

for l in onlyfiles:
    print(l)

x = "/Users/erdemdemir/Desktop/bird-dir/" + file_n[14] + ".txt"
print(x)

file_to_wr = open(x,"w+")

for c in onlyfiles:
    file_to_wr.write("{}\n".format(c))
file_to_wr.close()
'''


