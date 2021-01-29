import zipfile
import glob

i = 0
for filename in glob.iglob('C:\\Users\\Maria Sameen\\Desktop\\APTMalware-master\\samples\\**\\*.zip'):
    print(filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        i = i+1
        zip_ref.extractall('C:\\Users\\Maria Sameen\\Desktop\\extracted\\**\\'+str(i), pwd = bytes('infected', 'utf-8'))
