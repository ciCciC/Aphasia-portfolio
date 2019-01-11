
<h1>Transformer audio word to MFCCs</h1>
<p>Met deze transformer genereer ik dataset met daarin audio MFCC features op woord niveau en de gerelateerde woord. Deze dataset is voor om modellen mee te trainen.</p>


```python
import os, io, wave, csv, json, re, glob
import librosa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.cm as cm
import matplotlib as mpl
from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import lifter
from python_speech_features import logfbank
from python_speech_features import get_filterbanks
from python_speech_features import fbank
```


```python
# Voor het inlezen van bestanden uit een map.
def getFiles(folderpath):
    return glob.glob(folderpath)


# Voor het krijgen van de juiste sample tijd
def getTime(seconds, sample_rate):
    return int(seconds * sample_rate)


# Voor het exporteren van een data naar een CSV bestand.
def exportDataCSV(region, label, sample_rate, audiopath, writer):
    regionFeatures = '|'.join(['{:}'.format(x) for x in region[0].flatten()])
    writer.writerow({'region': regionFeatures, 'label': label, 'sample_rate': sample_rate, 
                     'begin':region[1], 'end':region[2], 'audiopath':audiopath})


# Voor het exporteren van data naar een CSV bestand.
def exportDatasCSV(regions, label, sample_rate, audiopath, writer):
    for region in regions:
        exportDataCSV(region, label, sample_rate, audiopath, writer)
```


```python
# Voor het krijgen van features van een audio signaal
def getSignalMFCC(signal, sample_rate, winlen, winstep, dct=True):
    mfcc_feat = 0
    if dct:
        mfcc_feat = mfcc(signal, sample_rate, winlen=winlen, winstep=winstep, nfft=512)
    else:
        mfcc_feat = mfccWithoutDCT(signal, sample_rate, winlen=winlen, winstep=winstep, nfft=512)
    return delta(mfcc_feat, 2)


# Een aangepaste variant van de MFCC methode waar de DCT methode niet wordt toegepast.
def mfccWithoutDCT(signal,samplerate=16000,winlen=0.010,winstep=0.001,numcep=13,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:np.ones((x,))):
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = np.log(feat)
#     feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy)
    return feat


# Deze methode maakt een data aan met de aangegeven interval binnen een begin en eind tijdsduur van een audiosegment
def getExtractedFeaturesWithInterval(audio_features, begin, end, ms, interval):
    begin = int(begin*ms)
    end = int(end*ms)
    # Audiosegment met de aangegeven begin en eindtijd
    word_features = audio_features[begin:end]
    return '|'.join(['{:}'.format(x) for x in [word_features[step:ms+step] for step in range(0, int(end-begin), ms)][0].flatten()])


# Voor het krijgen van tijdsduur van een signaal
def getAudioDuration(signal, sample_rate):
    return signal.shape[0] / float(sample_rate)
```


```python
# Map waar het CSV bestand opgeslagen moet worden
datasetDir = '/datb/aphasia/languagedata/corpus/dataset/'

# CSV bestand naam
# filename = 't_sne_word_st_dataset.csv'

# Map waar alle CSV bestanden voor de generator zijn opgeslagen
folderpath = '/datb/aphasia/languagedata/corpus/final/*'
```


```python
files = getFiles(folderpath)
```

<h3>De generator waarmee alleen de features van de woorden worden opgeslagen met de bijbehorende woord en karakters van het woord.</h3>


```python
ms = 1000

with open(datasetDir + filename, 'w') as toWrite:
    fieldnames = ['features', 'word', 'characters']
    writer = csv.DictWriter(toWrite, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=',')

    writer.writeheader()
    
    print('In progress...')
    
    for file in files:

        words_df = pd.read_csv(file)

        words = words_df.loc[words_df.word.str.startswith('st') | words_df.word.str.startswith('St')]

        if len(words) > 0:
            sample_rate, audio = wav.read(words.iloc[0, -1])
            audio_features = getSignalMFCC(audio, sample_rate, winlen=0.010, winstep=0.001, dct=True)

            for index, row in words.iterrows():
                word_features = getExtractedFeaturesWithInterval(audio_features, float(row['begin']), float(row['end']), ms, 5)
                writer.writerow({'features': word_features, 'word': row['word'], 'characters': list(row['word'])})

print('Finished')
```

    In progress...
    Finished


<p>Resultaat</p>


```python
filename = 't_sne_word_st_dataset.csv'
datasetDir = '/datb/aphasia/languagedata/corpus/dataset/'
df = pd.read_csv(datasetDir + filename, sep=',', skiprows=1, names=['features', 'word', 'characters'])

print(df.head(10))
```

                                                features              word  \
    0  0.19125565273940417|-1.2126446855308202|-1.337...  staatsgasbedrijf   
    1  0.11710806988029461|-0.8329364981486981|0.5362...            studio   
    2  0.06086896728795708|-1.3035405931899546|0.1037...           station   
    3  0.047536495415977244|-1.6194421262273486|0.809...             staan   
    4  0.1403275796144051|-0.6467594069001009|0.20157...             staan   
    5  0.11869705957178453|-0.3869094587441467|0.4103...             staan   
    6  -0.012332630803638977|0.9063030447378001|1.909...              stuk   
    7  0.1966308824837128|-0.9990856226072168|-2.0327...            steeds   
    8  0.10473927395167522|0.07471589035504848|-0.966...            stukje   
    9  0.15614319300190757|0.24638435863115973|1.3379...            steeds   
    
                                              characters  
    0  ['s', 't', 'a', 'a', 't', 's', 'g', 'a', 's', ...  
    1                     ['s', 't', 'u', 'd', 'i', 'o']  
    2                ['s', 't', 'a', 't', 'i', 'o', 'n']  
    3                          ['s', 't', 'a', 'a', 'n']  
    4                          ['s', 't', 'a', 'a', 'n']  
    5                          ['s', 't', 'a', 'a', 'n']  
    6                               ['s', 't', 'u', 'k']  
    7                     ['s', 't', 'e', 'e', 'd', 's']  
    8                     ['s', 't', 'u', 'k', 'j', 'e']  
    9                     ['s', 't', 'e', 'e', 'd', 's']  



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1988 entries, 0 to 1987
    Data columns (total 3 columns):
    features      1988 non-null object
    word          1988 non-null object
    characters    1988 non-null object
    dtypes: object(3)
    memory usage: 46.7+ KB


<h2>Transformer audio word zonder DCT transformatie.</h2>
<p>Zonder DCT transformatie want om een CNN te kunnen trainen heb ik meer informatie nodig van de features.</p>


```python
def getExtractedFeaturesNoInterval(audio_features, begin, end, ms, distance):
    begin = int((begin*ms)-distance)
    end = int(end*ms)
    word_features = audio_features[begin:end]
    return '|'.join(['{:}'.format(x) for x in word_features.flatten()])
```


```python
files = getFiles(folderpath)
print(f'Aantal files: {len(files)}')
```

    Aantal files: 742



```python
# woord_mfcc_Dir = '/datb/aphasia/languagedata/corpus/woord_mfcc/'
woord_mfcc_Dir = '/datb/aphasia/languagedata/corpus/woord_mfcc_zonder_dct/'
```


```python
ms = 1000

print('In progress...')
    
for file in files:
    filename = file.split('/')[-1]

    words_df = pd.read_csv(file)
    
    with open(woord_mfcc_Dir + filename, 'w') as toWrite:
        
        fieldnames = ['features', 'begin', 'end','word']
        writer = csv.DictWriter(toWrite, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=',')

        writer.writeheader()

        sample_rate, audio = wav.read(words_df.iloc[0, -1])
        audio_features = getSignalMFCC(audio, sample_rate, winlen=0.010, winstep=0.001, dct=False)

        for index, row in words_df.iterrows():
            word_features = getExtractedFeaturesNoInterval(audio_features, float(row['begin']), float(row['end']), ms)
            writer.writerow({'features': word_features, 'begin':row['begin'], 'end':row['end'], 'word': row['word']})

print('Finished')
```

    In progress...
    Finished


<p>Resultaat</p>


```python
result_files = getFiles(woord_mfcc_Dir+'*')
```


```python
print(f'Aantal CVs: {len(result_files)}')
```

    Aantal CVs: 742



```python
# word_df = pd.read_csv(result_files[0], sep=',', skiprows=1, names=['features', 'begin', 'end', 'word', 'characters']).drop(['characters'], axis=1)
print('Bestand: ' + result_files[0])
word_df = pd.read_csv(result_files[0], sep=',', skiprows=1, names=['features', 'begin', 'end', 'word'])
print(word_df.head(10))
```

    Bestand: /datb/aphasia/languagedata/corpus/woord_mfcc_zonder_dct/fn004915.csv
                                                features  begin    end  \
    0  0.06778698577608448|0.13087423249114744|-0.514...  0.478  0.910   
    1  0.007862618564192302|0.3674830396871556|-0.185...  0.910  1.143   
    2  0.001072975450130187|-0.5440577977407159|-1.84...  1.143  1.589   
    3  0.03390637301359121|-0.0008938322677181532|-0....  1.589  1.704   
    4  0.08841942234036537|-1.0428010846748221|-1.150...  1.704  2.450   
    5  -0.4122820757361211|1.3589979219353716|0.79978...  2.450  2.558   
    6  0.21904668743180764|0.7231182166591793|0.78785...  2.558  2.629   
    7  -0.12066839644772856|-0.5985963272300807|-0.46...  2.731  2.924   
    8  0.27790394494428095|-0.8738979498009506|0.9564...  2.924  3.182   
    9  -0.09673212523857444|0.045632403519313414|0.20...  3.182  3.322   
    
                word  
    0      tennisser  
    1         Raemon  
    2        Sluiter  
    3             is  
    4  uitgeschakeld  
    5             op  
    6            het  
    7          Dutch  
    8           Open  
    9             in  


<h2>Transformer audio word met DCT transformatie, 25ms windowsize en 10ms windowstep.</h2>
<p>Deze transformer transformeert de aangegeven tijd van een stukje audio naar MFCC met de aangegeven parameters zoals windowsize of windowstep. Vervolgens genereert hij een nieuwe dataset bestaande uit bijv. features, begin, end, word en fonemen. Deze transformers heb ik ontwikkeld voor Jeroen en Erik zodat zij een model kunnen trainen.</p>


```python
# Source data folder
folderpath = '/datb/aphasia/languagedata/corpus/final_fonemen/*'
```


```python
files = getFiles(folderpath)
print(f'Aantal files: {len(files)}')
```

    Aantal files: 742



```python
# Target folder
target_Dir = '/datb/aphasia/languagedata/corpus/woord_fonemen_mfcc/'
```


```python
ms = 100
distance = 10

print('In progress...')

fromfields = ['begin', 'end', 'word', 'wordtranscription','phonemes', 'audiopath']
tofields = ['features', 'begin', 'end', 'word', 'wordtranscription', 'phonemes']

for file in files:
    filename = file.split('/')[-1]

    words_df = pd.read_csv(file, sep=',', skiprows=1, names=fromfields)
    words_df = words_df[words_df.word != 'None']
    
    with open(target_Dir + filename, 'w') as toWrite:
        
        writer = csv.DictWriter(toWrite, fieldnames=tofields, quoting=csv.QUOTE_ALL, delimiter=',')

        writer.writeheader()

        sample_rate, audio = wav.read(words_df.iloc[0, -1])
        audio_features = getSignalMFCC(audio, sample_rate, winlen=0.025, winstep=0.01, dct=True)

        for index, row in words_df.iterrows():
            word_features = getExtractedFeaturesNoInterval(audio_features, float(row['begin']), float(row['end']), ms, distance)
            writer.writerow({tofields[0]: word_features, tofields[1]:row['begin'],
                             tofields[2]:row['end'], tofields[3]: row['word'],
                             tofields[4]:row['wordtranscription'], tofields[5]:row['phonemes']})

print('Finished')
```

    In progress...
    Finished


<p>Resultaat</p>


```python
result_files = getFiles(target_Dir+'*')
print(f'Target aantal files: {len(files)}')
print(f'Successful aantal CVs: {len(result_files)}')
```

    Target aantal files: 742
    Successful aantal CVs: 742



```python
ph_df = pd.read_csv(result_files[0], sep=',', skiprows=1, names=tofields)
ph_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>begin</th>
      <th>end</th>
      <th>word</th>
      <th>wordtranscription</th>
      <th>phonemes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.026312490791167063|0.7811919092575806|2.349...</td>
      <td>0.48</td>
      <td>0.91</td>
      <td>tennisser</td>
      <td>t|e|n|e|s|e|r</td>
      <td>t|E|n|@|s|@|r</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.06502826779564401|1.340519574243001|-0.53455...</td>
      <td>0.91</td>
      <td>1.14</td>
      <td>Raemon</td>
      <td>r|ee|m|o|n</td>
      <td>r|e|m|O|n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.15822498954374212|-0.26385485764938643|0.021...</td>
      <td>1.14</td>
      <td>1.59</td>
      <td>Sluiter</td>
      <td>s|l|ui|t|e|r</td>
      <td>s|l|Y+|t|@|r</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3720034613145941|2.0654791604140836|-3.34982...</td>
      <td>1.59</td>
      <td>1.70</td>
      <td>is</td>
      <td>i|s</td>
      <td>I|s</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.45984076899787424|-6.812778196047725|2.76429...</td>
      <td>1.70</td>
      <td>2.45</td>
      <td>uitgeschakeld</td>
      <td>ui|t|ch|e|s|ch|aa|k|e|l|t</td>
      <td>Y+|t|x|@|s|x|a|k|@|l|t</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.7485777578943896|0.756262421845205|3.6388619...</td>
      <td>2.45</td>
      <td>2.56</td>
      <td>op</td>
      <td>o|b</td>
      <td>O|b</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.09818803275224539|2.739280910271071|1.30769...</td>
      <td>2.56</td>
      <td>2.63</td>
      <td>het</td>
      <td>e|d</td>
      <td>@|d</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.12598375678069224|1.6479769014609633|2.0403...</td>
      <td>2.73</td>
      <td>2.92</td>
      <td>Dutch</td>
      <td>d|u|t|sj</td>
      <td>d|Y|t|S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.019180967955179|-11.411552545510702|-3.33577...</td>
      <td>2.92</td>
      <td>3.18</td>
      <td>Open</td>
      <td>oo|p|e|n</td>
      <td>o|p|@|n</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.7697786009411847|-2.838768484481993|-2.1564...</td>
      <td>3.18</td>
      <td>3.32</td>
      <td>in</td>
      <td>i|n</td>
      <td>I|n</td>
    </tr>
  </tbody>
</table>
</div>


