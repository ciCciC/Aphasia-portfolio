
<h1>Transforming Corpus data</h1>
<p>(c) Koray</p>
<p>In deze notebook transformeren we eerst de data naar de gewenste structuur. Daarna gebruiken we de "Phoneme Boundary Generator" om dataset te genereren voor neurale netwerken zoals MLP Classifier, LSTM etc.</p>


```python
import os, io, wave, csv, json, re, glob
from pydub import AudioSegment

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.cm as cm
import matplotlib as mpl

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import get_filterbanks
from python_speech_features import fbank
```

<h3>Benodigde methoden</h3>


```python
# Voor het inlezen van een dictionary of een CSV
def readDict(filepath):
    with open(filepath, 'r') as csvfile:
        return [row for row in csv.DictReader(csvfile)]

    
# Voor het inlezen van bestanden uit een map.
def getFiles(folderpath):
    files = glob.glob(folderpath + '*')
    return files


# Deze methode is om de hertz van een audio te transformeren naar de gewenste hertz
def transform_audio_hertz(audiofile, audiopath, extension, frame_rate):
    audiofile.set_frame_rate(frame_rate).set_channels(1).export(audiopath, format=extension)
    
    
# Voor het exporteren van data uit Corpus naar een CSV bestand.
def exportDataCSV(filepath, finalpath, fieldnames, uitfilteren):
    # map waarin de wav bestanden zijn opgeslagen
    wavFolderPath = '/datb/aphasia/languagedata/corpus/transform/wavfiles/'
    
    # map waarin de csv bestanden moeten opgeslagen worden
    finalPath = '/datb/aphasia/languagedata/corpus/'+finalpath
    
    with open(filepath, encoding = "ISO-8859-1") as toRead:
        read_data = toRead.readline()
        
        filename = filepath.split('/')[-1].split('.')[0]
        wavfile = wavFolderPath + filename + '.wav'
        
        with open(finalPath+filename+'.csv', 'w') as writeTo:
                # De structuur die van belang is voor de Phoneme boundary generator
                writer = csv.DictWriter(writeTo, fieldnames=fieldnames)
                writer.writeheader()

                count = 0
                subCount = 0

                begin = 0
                end = 0
                text = ''
                exclude = 'IntervalTier'

                while read_data:

                    if count > 11:
                        subCount += 1

                        if exclude in read_data:
                            break

                        if subCount == 1:
                            begin = float(read_data)
                        elif subCount == 2:
                            end = float(read_data)
                        elif subCount == 3:
                            text = re.sub('[\n\r"".?]', '', read_data)
                            
                            # Het uit filteren van de ongewenste data
                            if len(text) > 0 and not any(x in text for x in uitfilteren):
                                writer.writerow({'begin': begin, 'end': end, 'word': text, 'audiopath':wavfile})

                            begin = 0
                            end = 0
                            text = ''
                            subCount = 0

                    read_data = toRead.readline()
                    count += 1

# Methode decodeert phonemen naar woord en creert phonemen lijst. Die worden vervolgens returnd.
def phonemeDecoder(phonemeString, df_phsymbols):    
    count = 0
    phonemeChars = []
    word = []

    while count < len(phonemeString):
        currentChar = phonemeString[count]
        nextChar = ''

        if currentChar in df_phsymbols.encode.values:

            if int(count+1) < len(phonemeString):
                nextChar = currentChar+phonemeString[count+1]

            if nextChar in df_phsymbols.encode.values:
                tmpNext = df_phsymbols[df_phsymbols.encode == nextChar].values[0][0]
                phonemeChars.append(nextChar)
                word.append(tmpNext)
            else:
                tmp = df_phsymbols[df_phsymbols.encode == currentChar].values[0][0]
                phonemeChars.append(currentChar)
                word.append(tmp)

            nextChar = ''
        count += 1
        
    return word, phonemeChars
                    
# Voor het exporteren van fonemen uit Corpus naar een CSV bestand.
def exportFonemenCSV(filepath, folderfonemen, fieldnames, uitfilteren, df_phsymbols):
    # map waarin de wav bestanden zijn opgeslagen
    
    corpus_path = '/datb/aphasia/languagedata/corpus/'
    
    # map waarin de wav bestanden zijn opgeslagen
    wavFolderPath = corpus_path + 'transform/wavfiles/'
    
    #     Pad naar het al bestaande CSV bestand met daarin de woorden
    final_file_name = corpus_path + 'final/' + filepath.split('/')[-1].split('.')[0] + '.csv'
    
    #     Voor het inlezen van het al bestaande CSV bestand met daarin de woorden
    dict_finalfile = pd.read_csv(final_file_name, sep=',', skiprows=1, names=['begin', 'end', 'word', 'audiopath'])
    
    # map waarin de csv bestanden moeten opgeslagen worden
    final_fonemen_folder = corpus_path + folderfonemen
    
    with open(filepath, encoding = "ISO-8859-1") as toRead:
        read_data = toRead.readline()
        
        filename = filepath.split('/')[-1].split('.')[0]
        wavfile = wavFolderPath + filename + '.wav'
        
        with open(final_fonemen_folder+filename+'.csv', 'w') as writeTo:
            
                writer = csv.DictWriter(writeTo, fieldnames=fieldnames)
                writer.writeheader()

                count = 0
                subCount = 0
                countTier = 0

                begin = 0
                end = 0
                text = ''
                exclude = 'IntervalTier'

                while read_data:

                    if countTier == 2:

                        if count > 3:
                            subCount += 1

                            if exclude in read_data:
                                break

                            if subCount == 1:
                                begin = round(float(read_data), 2)
                            elif subCount == 2:
                                end = round(float(read_data), 2)
                            elif subCount == 3:
                                text = re.sub('[\n\r"".?]', '', read_data)

                                # Het uit filteren van de ongewenste data
                                if len(text) > 0 and not any(x in text for x in uitfilteren):
                                    word = ''
                                    wordtranscription, phonemeslist = phonemeDecoder(text, df_phsymbols)

                                    # Om exception te voorkomen bij niet bestaan van de waarde
                                    try:
                                        word = dict_finalfile[(dict_finalfile.begin.round(2) == begin) & (dict_finalfile.end.round(2) == end)].values[0][2]
                                    except IndexError:
                                        word = 'None'

                                    writer.writerow({fieldnames[0]: begin, fieldnames[1]: end, fieldnames[2]: word, fieldnames[3]:'|'.join(ph for ph in wordtranscription), 
                                                     fieldnames[4]:'|'.join(ph for ph in phonemeslist), fieldnames[5]:wavfile})

                                begin = 0
                                end = 0
                                text = ''
                                subCount = 0

                        count += 1
                    elif exclude in read_data:
                        countTier += 1

                    read_data = toRead.readline()
```

<h3>Eerst de wav bestanden converteren naar 16khz.</h3>
<p>Dit proces hoeft maar 1 keer uitgevoerd te worden.</p>


```python
folderpath = '/datb/aphasia/languagedata/corpus/transform/wavfiles/'

# Get all csv files where the audiopaths are saved
audiofiles = getFiles(folderpath)

notConvertable = []

# A batch for converting all Corpus audiofiles to a desired HERTZ which is 16000hz
for audiopath in audiofiles:
    try:
        audiofile = AudioSegment.from_file(audiopath, format="wav")
        transform_audio_hertz(audiofile, audiopath, 'wav', 16000)
    except Exception:
        notConvertable.append(audiopath)

print('Converting hertz to 16khz is finished')
```

    Converting hertz to 16khz is finished


<p>Hieronder wordt gekeken of de wav bestanden die niet kunnen worden geconverteerd naar 16khz afkomstig zijn van de map "comp-c". Dit heb ik kunnen realiseren door de naam van het bestand te gebruiken om te matchen met de bestaande non convertable wav bestanden in "notConvertable" lijst. Dit is de enige map waarin alleen 8khz wav bestanden in zitten en die hebben wij niet nodig.</p>


```python
folderpath = '/datb/aphasia/languagedata/corpus/original/wrd/comp-c/nl/'
wavPath = '/datb/aphasia/languagedata/corpus/transform/wavfiles/'

folderBroad = sc.parallelize(getFiles(folderpath))
numbers = folderBroad.filter(lambda x: wavPath+x.split('/')[-1].split('.')[0]+'.wav' in notConvertable).collect()
print('Non Convertable: {}'.format(len(notConvertable)))
print('Map:"comp-c" <- has 8khz wavfiles, match: {}'.format(len(numbers)))
```

    Non Convertable: 93
    Map:"comp-c" <- has 8khz wavfiles, match: 93


<h3>Batch voor het transformeren van de data naar de juiste structuur en opslaan in een CSV bestand.</h3>
<p>MFCC en woord</p>


```python
# waar we de mappen uit moeten halen
folderpath = '/datb/aphasia/languagedata/corpus/original/wrd/'

# sub map waar we de data uit moeten halen
subFolder = 'nl/'

# Mappen die we moeten uit filteren want die gaan we niet gebruiken
excludeFolders = ['n', 'm', 'l', 'i', 'a', 'c']

foldernames = list(filter(lambda x: not any(y in x[-1] for y in excludeFolders) , getFiles(folderpath)))

print('Folder names to use:')
print(foldernames)

print('---------------------')

for folder in foldernames:
    # ophalen van de bestanden per map
    files = getFiles(folder+'/'+subFolder)
    
    # de bestanden meegeven aan de "exportDataCSV" methode die de data op de juiste structuur zet en opslaat in een CSV bestand.
    for filepath in files:
        exportDataCSV(filepath, 'final/', ['begin', 'end', 'word', 'audiopath'], ['ggg', 'XXX', 'xxx', '...', '_', '*', '-'])
        
    print('Finished folder: {}',format(folder))
        
print('Finished')
```

    Folder names to use:
    ['/datb/aphasia/languagedata/corpus/original/wrd/comp-f', '/datb/aphasia/languagedata/corpus/original/wrd/comp-e', '/datb/aphasia/languagedata/corpus/original/wrd/comp-k', '/datb/aphasia/languagedata/corpus/original/wrd/comp-j', '/datb/aphasia/languagedata/corpus/original/wrd/comp-h', '/datb/aphasia/languagedata/corpus/original/wrd/comp-b', '/datb/aphasia/languagedata/corpus/original/wrd/comp-o', '/datb/aphasia/languagedata/corpus/original/wrd/comp-g']
    ---------------------
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-f
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-e
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-k
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-j
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-h
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-b
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-o
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-g
    Finished


<h3>Resultaat</h3>


```python
finalFolder = '/datb/aphasia/languagedata/corpus/final/*'
files = getFiles(finalFolder)

print('Aantal: {}'.format(len(files)))
print(files[0:10])
```

    Aantal: 742
    ['/datb/aphasia/languagedata/corpus/final/fn004915.csv', '/datb/aphasia/languagedata/corpus/final/fn002854.csv', '/datb/aphasia/languagedata/corpus/final/fn002059.csv', '/datb/aphasia/languagedata/corpus/final/fn003438.csv', '/datb/aphasia/languagedata/corpus/final/fn006104.csv', '/datb/aphasia/languagedata/corpus/final/fn007102.csv', '/datb/aphasia/languagedata/corpus/final/fn004375.csv', '/datb/aphasia/languagedata/corpus/final/fn000095.csv', '/datb/aphasia/languagedata/corpus/final/fn004851.csv', '/datb/aphasia/languagedata/corpus/final/fn007146.csv']



```python
exampleFile = '/datb/aphasia/languagedata/corpus/final/fn004915.csv'

# Using Pandas for reading dataset csv
file_df = pd.read_csv(exampleFile, sep=',', skiprows=1, names=['begin', 'end', 'word', 'audiopath'])

print('Example file: fn004915.csv')
print(file_df.head())
```

    Example file: fn004915.csv
       begin    end           word  \
    0  0.478  0.910      tennisser   
    1  0.910  1.143         Raemon   
    2  1.143  1.589        Sluiter   
    3  1.589  1.704             is   
    4  1.704  2.450  uitgeschakeld   
    
                                               audiopath  
    0  /datb/aphasia/languagedata/corpus/transform/wa...  
    1  /datb/aphasia/languagedata/corpus/transform/wa...  
    2  /datb/aphasia/languagedata/corpus/transform/wa...  
    3  /datb/aphasia/languagedata/corpus/transform/wa...  
    4  /datb/aphasia/languagedata/corpus/transform/wa...  


<h3>Batch voor het transformeren van de data naar de juiste structuur en opslaan in een CSV bestand.</h3>
<p>MFCC, woord en fonemen</p>


```python
# waar we de mappen uit moeten halen
folderpath = '/datb/aphasia/languagedata/corpus/original/wrd/'

# sub map waar we de data uit moeten halen
subFolder = 'nl/'

# Mappen die we moeten uit filteren want die gaan we niet gebruiken
excludeFolders = ['n', 'm', 'l', 'i', 'a', 'c']

foldernames = list(filter(lambda x: not any(y in x[-1] for y in excludeFolders) , getFiles(folderpath)))

print('Folder names to use:')
print(foldernames)

print('---------------------')

filesymbols = 'fonologie_format.csv'
df_phsymbols = pd.read_csv(filesymbols, sep=',', names=['decode', 'encode'])

print('In progress...')
for folder in foldernames:
    # ophalen van de bestanden per map
    files = getFiles(folder+'/'+subFolder)
    
    # de bestanden meegeven aan de "exportDataCSV" methode die de data op de juiste structuur zet en opslaat in een CSV bestand.
    for filepath in files:
        exportFonemenCSV(filepath, 'final_fonemen/',
                         ['begin', 'end', 'word', 'wordtranscription','phonemes', 'audiopath'],
                         ['ggg', '...', '_', '*', '-'],
                         df_phsymbols)

    print('Finished folder: {}',format(folder))
        
print('Finished')
```

    Folder names to use:
    ['/datb/aphasia/languagedata/corpus/original/wrd/comp-f', '/datb/aphasia/languagedata/corpus/original/wrd/comp-e', '/datb/aphasia/languagedata/corpus/original/wrd/comp-k', '/datb/aphasia/languagedata/corpus/original/wrd/comp-j', '/datb/aphasia/languagedata/corpus/original/wrd/comp-h', '/datb/aphasia/languagedata/corpus/original/wrd/comp-b', '/datb/aphasia/languagedata/corpus/original/wrd/comp-o', '/datb/aphasia/languagedata/corpus/original/wrd/comp-g']
    ---------------------
    In progress...
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-f
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-e
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-k
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-j
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-h
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-b
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-o
    Finished folder: {} /datb/aphasia/languagedata/corpus/original/wrd/comp-g
    Finished



```python
exampleFile = '/datb/aphasia/languagedata/corpus/final_fonemen/fn000007.csv'

# Using Pandas for reading dataset csv
file_df = pd.read_csv(exampleFile, sep=',', skiprows=1, names=['begin', 'end', 'word', 'wordtranscription','phonemes', 'audiopath'])

print('Example file: fn000007.csv')
print(file_df.head(11))
```

    Example file: fn000007.csv
        begin   end    word wordtranscription     phonemes  \
    0    0.00  0.20    None               NaN          NaN   
    1    0.20  0.74   zoals      z|oo|w|a|l|s  z|o|w|A|l|s   
    2    0.74  0.99    None                 e            @   
    3    0.99  1.20    None               a|l          A|l   
    4    1.20  1.57  gezegd     ch|e|z|e|ch|t  x|@|z|E|x|t   
    5    1.57  1.67      uh                 e            @   
    6    1.67  1.90      we               w|e          w|@   
    7    1.90  2.12    zijn            z|ij|n       z|E+|n   
    8    2.12  2.33     dan             d|a|n        d|A|n   
    9    2.33  2.50     toe              t|oe          t|u   
    10   2.50  2.81     aan              aa|n          a|n   
    
                                                audiopath  
    0   /datb/aphasia/languagedata/corpus/transform/wa...  
    1   /datb/aphasia/languagedata/corpus/transform/wa...  
    2   /datb/aphasia/languagedata/corpus/transform/wa...  
    3   /datb/aphasia/languagedata/corpus/transform/wa...  
    4   /datb/aphasia/languagedata/corpus/transform/wa...  
    5   /datb/aphasia/languagedata/corpus/transform/wa...  
    6   /datb/aphasia/languagedata/corpus/transform/wa...  
    7   /datb/aphasia/languagedata/corpus/transform/wa...  
    8   /datb/aphasia/languagedata/corpus/transform/wa...  
    9   /datb/aphasia/languagedata/corpus/transform/wa...  
    10  /datb/aphasia/languagedata/corpus/transform/wa...  

