import os

path='data/obj/'


imgList=os.listdir(r'D:\Doktorat_zajecia\CV\PROJECT\cropped do uczenia\test')

print(imgList)

textFile=open('test.txt','w')

for img in imgList:
    imgPath=path+img+'\n'
    textFile.write(imgPath)
