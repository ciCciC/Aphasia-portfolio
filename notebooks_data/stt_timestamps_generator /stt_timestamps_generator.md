
<h1>Script for getting boundaries (begin and endtime boundaries of each word of each sentence)</h1>
<p>(c) Koray</p>
<p>Deze script is zeer van belang. Hem hebben we gebruikt om te kijken of Google goed is in het genereren van timestamps en voor het genereren van de timestamps van de woorden in de gewenste structuur. Dus met deze script kunnen we de gewenste structuur met timestamps van woorden genereren welke we vervolgens nodig hebben voor de Phoneme Boundary Generator.</p>


```python
# Get needed libraries
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.oauth2 import service_account

import os, io, wave, csv, json, re, glob
from ipywidgets import FloatProgress
from IPython.display import display

from pydub import AudioSegment
```

<h3>Create a class for getting the Credentials</h3>
<p>Credentials klassen om duplicatie te voorkomen. Om netter te programmeren.</p>


```python
# Create a class for getting the Credentials
class Credentials(object):

    file = os.getcwd()

    @staticmethod
    def getCredentials():
        return service_account.Credentials.from_service_account_file(Credentials.__serviceCredentials())
    
    @staticmethod
    def __serviceCredentials():
        return Credentials.file + '/audiototext-c92821bf0af8.json'
    
    def readjson():
        with open(Credentials.__serviceCredentials()) as json_data:
            d = json.load(json_data)
            print(d)
```

<h3>Create a class for configuring the audio for processing</h3>
<p>ConfigAudio klassen om duplicatie te voorkomen. Om netter te programmeren.</p>


```python
# Create a class for configuring the audio for processing
class ConfigAudio(object):
    audiopath = ''
    csvPath = ''
    filename = ''
    originwords = ''
    hertz = ''
    languageCode = ''
    testAudio = '/datb/aphasia/languagedata/voxforge/original/Kaas-20131123-tlz/wav/nl-0119.wav'
    testSentence = 'PAUL DEED DE BOVENSTE KNOOP VAN ZIJN OVERHEMD OPEN'

    def __init__(self, audiopath, originwords, csvPath, languageCode, hertz=None, demo=False):
        if demo:
            self.audiopath = self.testAudio
        else:
            self.audiopath = audiopath
            self.originwords = originwords
            self.csvPath = csvPath
            
        self.filename = audiopath.split('/')[-1].split('.')
        self.hertz = hertz
        self.languageCode = languageCode
```

<h3>Create a class for communication with the Google STT API</h3>
<p>AudioTranscribe klassen om duplicatie te voorkomen. Om netter te programmeren.</p>


```python
class AudioTranscribe(object):
    
    # Method for getting begin and end time of a word in a audio signal and saving in a csv file
    @staticmethod
    def GoogleSpeechToWords(ConfigAudio, finalDir):
        file_path = ConfigAudio.audiopath
        file_name = ConfigAudio.filename
        

        # Loads the audio into memory
        with io.open(file_path, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)

        # Init a client
        client = speech.SpeechClient(credentials=Credentials.getCredentials())

        # Init recognition configuration
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=ConfigAudio.hertz,
            language_code=ConfigAudio.languageCode,
            enable_word_time_offsets=True,
            speech_contexts=[speech.types.SpeechContext(phrases=ConfigAudio.originwords)]) #Give origin words to Google

        # start operation for getting timestamps
        operation = client.long_running_recognize(config, audio)

        # print('Waiting for operation to complete...')
        result = operation.result(timeout=90)

        # Save the boundaries, words, filepaths in a csv file
        for result in result.results:
            try:
                alternative = result.alternatives[0]
                AudioTranscribe.save__in__dict(file_name, alternative.words, finalDir, ConfigAudio.originwords, file_path)
            except IndexError:
                AudioTranscribe.save__in__dict(file_name, 'empty', finalDir, ConfigAudio.originwords, file_path)
            
      
    # Method for saving the words data in a csv
    @staticmethod
    def save__in__dict(filename, words, finalDir, originwords, audiopath):
        with open(finalDir+filename[0]+'.csv', 'w') as jsonfile:
            fieldnames = ['begin', 'end', 'word', 'audiopath']

            writer = csv.DictWriter(jsonfile, fieldnames=fieldnames)

            writer.writeheader()
            
            if 'empty' in words:
                empty = 'nan'
                for originword in originwords:
                    writer.writerow({'begin': empty,
                                     'end': empty,
                                     'word': originword,
                                     'audiopath': audiopath})
            else:
                for index, word_info in enumerate(words):
                    word = word_info.word
                    start_time = word_info.start_time
                    end_time = word_info.end_time

                    try:
                        originword = originwords[index]
                    except IndexError:
                        originword = 'nan'

                    writer.writerow({'begin': start_time.seconds + start_time.nanos * 1e-9,
                                     'end': end_time.seconds + end_time.nanos * 1e-9,
                                     'word': originword,
                                     'audiopath': audiopath})


    # Method for printing the results gathered from Google
    def printResult(words, originwords, file_path):
        for index, word_info in enumerate(words):
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            
            originword = originwords[index]
            
            print('Word: {}, start_time: {}, end_time: {}, filepath:{}'.format(
                    originword,
                    start_time.seconds + start_time.nanos * 1e-9,
                    end_time.seconds + end_time.nanos * 1e-9,
                    file_path))

```

<h3>Create needed methods for reading CSV files and extracting preferred amount files</h3>


```python
# For reading dictionary
def readDict(filepath):
    with open(filepath, 'r') as csvfile:
        return [sentence for sentence in csv.DictReader(csvfile)]
        
# Method for getting files with a desired amount and from a desired index
def getFiles(folderpath, amount=None, fromIndex=None):
    files = glob.glob(folderpath + '*')
    size = len(files)
    return files[fromIndex if fromIndex is not None else 0 : amount if amount is not None else size]
```

<h1>Transform text data into csv files</h1>
<p>This will make extracting data much easier for our GoogleSpeechToWords method of AudioTranscribe class.</p>
<p>This has to be executed only ones to do the transformation.</p>


```python
# Inlezen van alle bestanden
def getFiles(directory, name):
    subDir = '/etc/PROMPTS'
    return [glob.glob(directory+name), subDir]

# Transformeren van de text data in de gewenste structuur
def transformTextFile(folder, filepath):
    transformDir = '/datb/aphasia/languagedata/voxforge/transform/align/'
    filename = folder.split('/')[-1]
    
    with open(folder+filepath, 'r', encoding="ISO-8859-1") as readfile:
        line = readfile.readline()
        
        with open(transformDir+filename+'.csv', 'w') as writeTo:
                fieldnames = ['audiopath', 'sentence', 'words']
                writer = csv.DictWriter(writeTo, fieldnames=fieldnames)
                writer.writeheader()

                while line:
                    splitted = line.split(' ')
                    audiopath = splitted[0].replace('mfc', 'wav')+'.wav'
                    audiopath = '/datb/aphasia/languagedata/voxforge/original/'+audiopath
                    sentence = re.sub('[\n]', '', ' '.join(splitted[1:]).lower())
                    words = list(map(lambda x: re.sub('[\n]', '', x).lower(), splitted[1:]))

                    writer.writerow({'audiopath': audiopath,
                                     'sentence': sentence,
                                     'words': words})

                    line = readfile.readline()

                    
textfiles = getFiles('/datb/aphasia/languagedata/voxforge/original/', '*')
f = FloatProgress(min=0, max=len(textfiles), description='Restructuring:', bar_style='success', orientation='horizontal')
display(f)
                    
for textfile in textfiles[0]:
    transformTextFile(textfile, textfiles[1])
    f.value += 1
    
print('finished')
```

<p>De gewenste structuur csv bestand die vervolgens nodig is om mee te kunnen geven aan de batch voor het verkrijgen van de timestamps mbv Google STT service. Hieronder zie je de eindresultaat van de methode "transformTextFile()" die we hebben gebruikt om de ruwe data een gewenste structuur toe te kennen. Hieronder wordt 1 csv bestand als voorbeeld ingelezen om de structuur te weergeven.</p>


```python
import pandas as pd

pathAlignDir = '/datb/aphasia/languagedata/voxforge/transform/align/'
example_file = 'boergait-20080827-tks.csv'

# Using Pandas for reading dataset csv
df_dataForGoogle = pd.read_csv(pathAlignDir + example_file, sep=',', skiprows=1, names=['audiopath', 'sentence', 'words'])

print('Head:')
print(df_dataForGoogle.head())
print('----------------')
print('Tail:')
print(df_dataForGoogle.tail())
```

    Head:
                                               audiopath  \
    0  /datb/aphasia/languagedata/voxforge/original/b...   
    1  /datb/aphasia/languagedata/voxforge/original/b...   
    2  /datb/aphasia/languagedata/voxforge/original/b...   
    3  /datb/aphasia/languagedata/voxforge/original/b...   
    4  /datb/aphasia/languagedata/voxforge/original/b...   
    
                                                sentence  \
    0  in deze branche zijn lange werkdagen heel gebr...   
    1  de stichting werd niet-ontvankelijk verklaard ...   
    2  donderdag tot en met zondag niet warmer dan ze...   
    3  het jongste broertje is bepaald niet op zijn m...   
    4  wat er gebeurd is had in feite niets te betekenen   
    
                                                   words  
    0  ['in', 'deze', 'branche', 'zijn', 'lange', 'we...  
    1  ['de', 'stichting', 'werd', 'niet-ontvankelijk...  
    2  ['donderdag', 'tot', 'en', 'met', 'zondag', 'n...  
    3  ['het', 'jongste', 'broertje', 'is', 'bepaald'...  
    4  ['wat', 'er', 'gebeurd', 'is', 'had', 'in', 'f...  
    ----------------
    Tail:
                                               audiopath  \
    5  /datb/aphasia/languagedata/voxforge/original/b...   
    6  /datb/aphasia/languagedata/voxforge/original/b...   
    7  /datb/aphasia/languagedata/voxforge/original/b...   
    8  /datb/aphasia/languagedata/voxforge/original/b...   
    9  /datb/aphasia/languagedata/voxforge/original/b...   
    
                                                sentence  \
    5  de advocaat maakte creatief gebruik van het pr...   
    6  de grote meesters hangen in de mooiste zaal va...   
    7  voorafgaand aan de sloop heeft men het gebouw ...   
    8  toen iedereen de moed had opgegeven kwam esthe...   
    9             hij schudde de oplossing uit zijn mouw   
    
                                                   words  
    5  ['de', 'advocaat', 'maakte', 'creatief', 'gebr...  
    6  ['de', 'grote', 'meesters', 'hangen', 'in', 'd...  
    7  ['voorafgaand', 'aan', 'de', 'sloop', 'heeft',...  
    8  ['toen', 'iedereen', 'de', 'moed', 'had', 'opg...  
    9  ['hij', 'schudde', 'de', 'oplossing', 'uit', '...  


<h3>Batch for generating word boundaries</h3>
<p>Script below runs a batch for generating word boundaries and saving that including the filepath in a json file in the subfolder 'Final' of the folder 'VoxForge'. This has to be run only once!</p>


```python
finalDir = '/datb/aphasia/languagedata/voxforge/final/'

folderpath = '/datb/aphasia/languagedata/voxforge/transform/align/'

# Amount is the amount files (which contains sentences) to generate data from
files = getFiles(folderpath, amount=1)

f = FloatProgress(min=0, max=len(files), description='In progress:', bar_style='success', orientation='horizontal')
display(f)

# This is the batch which runs all sentences. E.G. 600 sentences.
for filepath in files:
    reader = readDict(filepath)

    for read in reader:
        
        cleanWords = list(map(lambda word: re.sub('[][]', '', word).strip(), read['words'].split(',')))
        audioPath = read['audiopath']
        sentence = read['sentence']

        AudioTranscribe.GoogleSpeechToWords(ConfigAudio(audiopath=audioPath,
                                                        originwords=cleanWords,
                                                        csvPath=filepath,
                                                        languageCode='nl-NL'), finalDir)
        f.value += 1
        
print('finished')
```

<p>Hieronder zie je de resultaat van de batch (van hierboven) die is uitgevoerd. Voor weergave van de structuur van het csv bestand wordt 1 csv bestand ingelezen. Deze data (dus alle csv bestanden) met de timestamps en woorden zijn uiteindelijk nodig voor de <b>Phoneme Boundary Generator!</b>. Want die generator gaat de benodigde dataset genereren om de Phoneme Boundary Classifier te kunnen trainen.</p>


```python
finalDir = '/datb/aphasia/languagedata/voxforge/final/'
example_file = 'nl-0027.csv'

# Using Pandas for reading dataset csv
df_data = pd.read_csv(finalDir + example_file, sep=',', skiprows=1, names=['begin', 'end', 'word', 'audiopath'])

print('Head:')
print(df_data.head())
print('----------------')
print('Tail:')
print(df_data.tail())
```

    Head:
       begin  end          word                                          audiopath
    0    0.5  0.9         'een'  /datb/aphasia/languagedata/voxforge/original/b...
    1    0.9  1.3  'vergunning'  /datb/aphasia/languagedata/voxforge/original/b...
    2    1.3  1.4          'in'  /datb/aphasia/languagedata/voxforge/original/b...
    3    1.4  1.5          'de'  /datb/aphasia/languagedata/voxforge/original/b...
    4    1.5  1.7         'zin'  /datb/aphasia/languagedata/voxforge/original/b...
    ----------------
    Tail:
        begin   end        word                                          audiopath
    6    1.85  2.15      'deze'  /datb/aphasia/languagedata/voxforge/original/b...
    7    2.15  2.30       'wet'  /datb/aphasia/languagedata/voxforge/original/b...
    8    2.45  2.70        'is'  /datb/aphasia/languagedata/voxforge/original/b...
    9    2.70  2.90      'geen'  /datb/aphasia/languagedata/voxforge/original/b...
    10   2.90  3.45  'vereiste'  /datb/aphasia/languagedata/voxforge/original/b...


<h3>This method runs a batch for separating the files in the folders with the usernames.</h3>
<p>This is needed for separating the same task to more users. For example: 3 users want to manual adjust de timestamps of the words in huge amount csv files, than you dont want to work in the same folder. So this is avoiding adjusting files which were already adjusted.</p>


```python
# This method runs a batch for separating the files in the folders with the usernames
def runBatch(users, files, generalDir, fromUser):
    
    f = FloatProgress(min=0, max=len(users), description='In progress:', bar_style='success', orientation='horizontal')
    display(f)
    
    jump= 20    
    userIndex = fromUser
    
    for xRange in range(fromUser*jump, len(files), jump):
        
        user = users[userIndex]
        
        for filepathIndex in range(xRange, xRange+jump):
            filepath = files[filepathIndex]
            reader = readDict(filepath)

            for read in reader:

                cleanWords = list(map(lambda word: re.sub('[][]', '', word).strip(), read['words'].split(',')))
                audioPath = read['audiopath']
                sentence = read['sentence']

                AudioTranscribe.GoogleSpeechToWords(ConfigAudio(audiopath=audioPath,
                                                                originwords=cleanWords,
                                                                csvPath=filepath,
                                                                languageCode='nl-NL'), generalDir+user+'/')
        userIndex += 1
        f.value += 1
    
```

<h3>Batch</h3>


```python
# This batch separates the csv data in the folder Align and saves them in the folders with the usernames.
# This is been executed with a batch

generalDir = '/datb/aphasia/languagedata/voxforge/cleanerTeam/'
folderpath = '/datb/aphasia/languagedata/voxforge/transform/align/'

users = ['koray', 'jesse', 'erik']

files = getFiles(folderpath, amount=61, fromIndex=1)

print(len(files))

runBatch(users, files, generalDir, fromUser=1)

print('finished')
```

    60



    FloatProgress(value=0.0, bar_style='success', description='In progress:', max=3.0)


    finished


<p>Hieronder zie je de resultaat van de batch (van hierboven) die is uitgevoerd. Voor weergave van de structuur van de mappen waarin de CSV bestanden zijn onderverdeeld zodat ieder kan beginnen met handmatig aanpassen van de timestamps van de woorden in de CSV bestanden. Dit is nodig om fouten veroorzaakt door Google te corrigeren om zo correct en accuraat mogelijke data te kunnen geven voor de stap: Phoneme Boundary Generator.</p>


```python
generalDir = '/datb/aphasia/languagedata/voxforge/cleanerTeam/'

koray_data = getFiles(generalDir+'koray/')
jesse_data = getFiles(generalDir+'jesse/')
erik_data = getFiles(generalDir+'erik/')

print('Koray data')
print(koray_data[0:10])

print('\nJesse data')
print(jesse_data[0:10])

print('\nErik data')
print(erik_data[0:10])
```

    Koray data
    ['/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0920.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0074.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0873.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0553.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0534.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0251.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0637.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0097.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0923.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/koray/nl-0204.csv']
    
    Jesse data
    ['/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0688.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0074.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0493.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0494.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0491.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0820.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0747.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0097.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0685.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/jesse/nl-0442.csv']
    
    Erik data
    ['/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0405.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0836.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0832.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0402.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0197.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0534.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0057.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0422.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0839.csv', '/datb/aphasia/languagedata/voxforge/cleanerTeam/erik/nl-0454.csv']



```python

```


```python

```
