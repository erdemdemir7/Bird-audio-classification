import os
from functions import playAudio
from functions import plotAudio
from functions import initiate_birds
from functions import initiate_libr

#Initials
path = os.getcwd() + "/"
song = os.getcwd() + "/bird-sounds/"
bird_path = path + "bird-dir/bird-types.txt"
bird_names = open(bird_path, "r")

#Birds List
birds = initiate_birds()

#Birds and their songs Dictionary
libr = initiate_libr(birds)


#Bird types list to choose from
tmp = list()
for x in libr.keys():
    tmp.append(x)

#####
# Bird types to choose from 0 to 14
t = tmp[2]

# ['azrm rep and soundfiles', 'bagm rep and soundfiles', 'bgmr rep and soundfiles',
# 'bgmy partial rep and soundfiles', 'bhmb prelim rep display and soundfiles',
# 'bmgw rep and soundfiles', 'bmor display and soundfiles done',
# 'bomp rep and soundfiles', 'boom rep and soundfiles', 'caim rep and soundfiles',
# 'ccym rep and soundfiles', 'cowm rep and soundfiles', 'cyom rep and soundfiles',
# 'ggmr rep and soundfiles', 'ggrm rep and soundfiles']

#Play ond plot bird songs counter # of times
counter = 10

for x in libr[t]:
    if counter == 0:
        break
    pth = song + t +"/" + x
    playAudio(pth) # Play song
    plotAudio(pth) # Plot song
    counter -= 1


'''
#To print the dictionary
for keys in libr.keys():
    print("keys:{}:{} {}\n".format(keys, len(libr[keys]),libr[keys]))
'''








