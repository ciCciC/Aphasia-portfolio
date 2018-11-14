
<h1>Phoneme boundary data generator</h1>
<p>(c) Koray </p>
<p>This algorithm generates dataset for training a phoneme boundary classifier</p>


```python
import os, io, wave, csv, json, re, glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.cm as cm
import matplotlib as mpl
from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import get_filterbanks
from python_speech_features import fbank
```

<h3>Benodigde methoden</h3>
<p>Deze methoden zijn van belang voor het uitvoeren van het proces. Elke methode heeft een eigen beschrijving van zijn functie.</p>


```python
# Voor het inlezen van een dictionary bestand.
def readDict(filepath):
    with open(filepath, 'r') as csvfile:
        return [sentence for sentence in csv.DictReader(csvfile)]

    
# Deze methode is om de hertz van een audio te transformeren naar de gewenste hertz
def transform_audio_hertz(audiofile, audiopath, extension, frame_rate):
    audiofile.set_frame_rate(frame_rate).set_channels(1).export(audiopath, format=extension)


# Voor het inlezen van bestanden uit een map.
def getFiles(folderpath, amount=None):
    files = glob.glob(folderpath + '*')
    size = len(files)
    return files[0:amount if amount is not None else size]


# Voor het krijgen van de juiste sample tijd
def getTime(seconds, sample_rate):
    return int(seconds * sample_rate)


# Methode om de audiosegmenten uit de regios te kunnen krijgen.
def getRegions(audio, side, boundary, frame_size, times, sample_rate):
    leftRegion = []
    rightRegion = []

    if 'L' in side:
        for walk in range(0, times):
            frame = boundary - (frame_size * walk)
            left = getTime(frame - frame_size, sample_rate)
            right = getTime(frame, sample_rate)
            tmpRegion = audio[left:right]
            leftRegion.append(tmpRegion)

    if 'R' in side:
        for walk in range(0, times):
            frame = boundary + (frame_size * walk)
            left = getTime(frame, sample_rate)
            right = getTime(frame + frame_size, sample_rate)
            tmpRegion = audio[left:right]
            rightRegion.append(tmpRegion)

    return leftRegion if 'L' in side else rightRegion


# Methode om de features uit de regios te kunnen krijgen.
def getRegionsFeatures(features_mfcc, side, boundary, frame_size, times):
    leftRegion = []
    rightRegion = []

    if 'L' in side:
        for walk in range(0, times):
            frame = boundary - (frame_size * walk)
            left = frame - frame_size
            right = frame
            tmpRegion = features_mfcc[left:right]
            leftRegion.append(tmpRegion)

    if 'R' in side:
        for walk in range(0, times):
            frame = boundary + (frame_size * walk)
            left = frame
            right = frame + frame_size
            tmpRegion = features_mfcc[left:right]
            rightRegion.append(tmpRegion)

    return leftRegion if 'L' in side else rightRegion


# Voor het exporteren van een data naar een CSV bestand.
def exportDataCSV(region, label, sample_rate, writer):
    region = '|'.join(['{:}'.format(x) for x in region.flatten()])
    writer.writerow({'region': region, 'label': label, 'sample_rate': sample_rate})

    
# Voor het exporteren van data naar een CSV bestand.
def exportDatasCSV(regions, label, sample_rate, writer):
    for region in regions:
        exportDataCSV(region, label, sample_rate, writer)
```


```python
# Voor het krijgen van features van een audio signaal
def getSignalMFCC(signal, sample_rate):
    mfcc_feat = mfcc(signal, sample_rate, winlen=0.010, winstep=0.001, nfft=512, ceplifter=22)
    return delta(mfcc_feat, 2)


# Een aangepaste variant van de MFCC methode waar de lifter methode niet wordt toegepast.
def getAdjustedMFCC(signal,samplerate=16000,winlen=0.010,winstep=0.001,numcep=13,
         nfilt=26,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:np.ones((x,))):

    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
#     feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy)
    return feat


# Voor het krijgen van tijdsduur van een signaal
def getAudioDuration(signal, sample_rate):
    return signal.shape[0] / float(sample_rate)


# Methode voor het snijden van features in meerdere dimensies 5,13
def transform2DFeatures(features, slice_boundary):
    transformed = []
    
    for x in range(0, len(features), slice_boundary):
        transformed.append([features[y] for y in range(x, x+slice_boundary)])
    
    return np.array(transformed)


# Voor het plotten van een audio signaal
def plotSignal(signal, sample_rate, features, figurNum, title, ylabel_title):
    fig = plt.figure(figurNum)
    T = getAudioDuration(signal, sample_rate)
    ax = fig.add_subplot(111)
    ax.imshow(np.flipud(features.T), cmap=cm.jet, aspect=0.08, extent=[0,T,0,13])
    ax.set_title(title, fontsize=26)
    ax.set_ylabel(ylabel_title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16)
    print('Frams [0] and filterbanks [1]' + str(features.shape))
    
```

<h3>Eerst de hertz van de audio converteren naar de gewenste hertz</h3>
<p>Dit onderdeel is van belang voor de audio naar MFCC transformatie voor het krijgen van de features</p>
<p>Dit hoeft maar 1x uitgevoerd te worden!</p>


```python
folderpath = '/datb/aphasia/languagedata/voxforge/transform/align/'

# Get all csv files where the audiopaths are saved
files = getFiles(folderpath)

# First check if all audio is in the format WAV before transforming to another HERTZ
count = 0
for fileIndex in range(0, len(files)):
    for audio in readDict(files[fileIndex]):
        audiopath = audio['audiopath'].split('/')[-1]
        count += 1 if 'wav' not in audiopath else 0

print('Amount of non-wav files: {}'.format(count))
print('Lookup is finished')


# A batch for converting all VoxForge audiofiles to a desired HERTZ which is 16000hz
if(count == 0):
    for fileIndex in range(0, len(files)):
        for audio in readDict(files[fileIndex]):
            try:
                audiofile = AudioSegment.from_wav(audio['audiopath'])
                transform_audio_hertz(audiofile, audio['audiopath'], 'wav', 16000)
            except FileNotFoundError:
                separated = audio['audiopath'].split('/')
                newName = re.sub('ï»¿', '',separated[-1].lower())
                newAudioPath = '/'.join(separated[:-1]) + '/' + newName
                audiofile = AudioSegment.from_wav(newAudioPath)
                transform_audio_hertz(audiofile, audio['audiopath'],'wav', 16000)

    print('Converting hertz is finished')
else:
    print('There are non-wav files!')
```

    Amount of non-wav files: 0
    Lookup is finished
    Converting hertz is finished


<h3>Voorbeeld melfilter en mfcc van een hele audio signaal.</h3>


```python
audioKoray = '/home/15068145/notebooks/test_koray/koray_woorden.wav'
# audioKoray = '/home/15068145/notebooks/test_koray/F60E2VT8.wav'

sample_rate, signal = wav.read(audioKoray)
# signal = signal[0:int(3.5 * sample_rate)]
features_log = logfbank(signal, sample_rate, nfft=512, nfilt=40, winlen=0.010, winstep=0.001)
features_mfcc = getSignalMFCC(signal, sample_rate)

plotSignal(signal=signal, sample_rate=sample_rate, features=features_log, figurNum=1, title='log melfilter', ylabel_title='Frequency (kHz)')
plotSignal(signal=signal, sample_rate=sample_rate, features=features_mfcc, figurNum=2, title='mfcc', ylabel_title='MFCC Coefficients')
plt.tight_layout()
plt.show()
```

    Frams [0] and filterbanks [1](8815, 40)
    Frams [0] and filterbanks [1](8815, 13)



![png](output_8_1.png)



![png](output_8_2.png)


<h3>[Versie 1, Generator] - Oude manier van genereren van trainings data. Zonder eerst de audio te transformeren naar MFCC.</h3>


```python
datasetDir = '/datb/aphasia/languagedata/voxforge/dataset/'

folderpath = '/datb/aphasia/languagedata/voxforge/final/'

# Get all csv files
files = getFiles(folderpath)

subRegion = 0.020
tsubRegion = subRegion / 2
region = 0.050

# Save dataset in a csv file
with open(datasetDir + 'datasetboundary.csv', 'w') as toWrite:

    fieldnames = ['region', 'label', 'sample_rate']
    writer = csv.DictWriter(toWrite, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=',')

    writer.writeheader()

    for x in range(0, len(files)):

        filedict = readDict(files[x])
        audiopath = filedict[0]['audiopath']

        audio, sample_rate = librosa.load(audiopath)

        count = 1
        while count < len(filedict):
            # Get prev and current word element
            prevW = filedict[count - 1]
            currW = filedict[count]
            
            # Get prev end-time and current begin-time
            boundaryL = float(prevW['end'])
            boundaryR = float(currW['begin'])

            # Get (true) left and right subregion segment
            tsubRegionL = audio[getTime(boundaryL-tsubRegion, sample_rate):getTime(boundaryL, sample_rate)]
            tsubRegionR = audio[getTime(boundaryR, sample_rate):getTime(boundaryR + tsubRegion, sample_rate)]

            # Get (false) subregions from left and right
            nRegionL = getRegions(audio, 'L', boundaryL - tsubRegion, subRegion, 3, sample_rate)
            nRegionR = getRegions(audio, 'R', boundaryR + tsubRegion, subRegion, 3, sample_rate)

            # Concatenate (true) left subregion and right subregion to ONE True region
            tRegion = np.concatenate((tsubRegionL, tsubRegionR), axis=None)
            
            tRegionFeatures = getSignalFeatures(tRegion, sample_rate)
            
            nRegionLfeatures = [getSignalFeatures(regionL, sample_rate) for regionL in nRegionL]
            nRegionRfeatures = [getSignalFeatures(regionR, sample_rate) for regionR in nRegionR]
            
            # Export to CSV
            exportDataCSV(tRegionFeatures, 1, sample_rate, writer)

            exportDatasCSV(nRegionLfeatures, 0, sample_rate, writer)
            exportDatasCSV(nRegionRfeatures, 0, sample_rate, writer)

            count += 1

print('finished')
```

<h3>[Versie 2, Generator] - Nieuwe manier van genereren van trainingsdata. Met eerst transformeren van audio naar MFCC.</h3>


```python
datasetDir = '/datb/aphasia/languagedata/voxforge/dataset/'

folderpath = '/datb/aphasia/languagedata/voxforge/final/'

# Get all csv files
files = getFiles(folderpath)

multiply_ms = int(1000)
subRegion = int(10)
tsubRegion = int(subRegion / 2)
size_region = 5

# Save dataset in a csv file
with open(datasetDir + 'datasetboundary.csv', 'w') as toWrite:

    fieldnames = ['region', 'label', 'sample_rate']
    writer = csv.DictWriter(toWrite, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=',')

    writer.writeheader()

#     for x in range(0, len(files)-(len(files)-1), 1):
    for x in range(0, len(files)):

        filedict = readDict(files[x])
        audiopath = filedict[0]['audiopath']
        
#         Read audio
        sample_rate, audio = wav.read(audiopath)
        print('Audio duration: {}, rate:{}'.format(getAudioDuration(audio, sample_rate), sample_rate))
        
#         Transform audio to mfcc to get features
        features_mfcc = getSignalMFCC(audio, sample_rate)

        count = 1
        while count < len(filedict):
            # Get prev and current word element
            prevW = filedict[count - 1]
            currW = filedict[count]
            
            # Get prev end-time and current begin-time
            boundaryL = int(float(prevW['end']) * multiply_ms)
            boundaryR = int(float(currW['begin']) * multiply_ms)

#             # Get (true) left and right subregion frames
            tsubRegionL = features_mfcc[boundaryL-tsubRegion:boundaryL]
            tsubRegionR = features_mfcc[boundaryR:boundaryR + tsubRegion]

#             # Get (false) subregions from left and right
            nRegionLfeatures = getRegionsFeatures(features_mfcc, 'L', boundaryL - tsubRegion, subRegion, size_region)
            nRegionRfeatures = getRegionsFeatures(features_mfcc, 'R', boundaryR + tsubRegion, subRegion, size_region)            
        
            # Concatenate (true) left subregion and right subregion to ONE True region
            tRegionFeatures = np.concatenate((tsubRegionL, tsubRegionR), axis=None)
            
#             # Export to CSV
            exportDataCSV(tRegionFeatures, 1, sample_rate, writer)

            exportDatasCSV(nRegionLfeatures, 0, sample_rate, writer)
            exportDatasCSV(nRegionRfeatures, 0, sample_rate, writer)

            count += 1

print('finished')
```

    Audio duration: 3.7546875, rate:16000
    Audio duration: 2.7306875, rate:16000
    Audio duration: 3.4986875, rate:16000
    Audio duration: 2.7306875, rate:16000
    Audio duration: 3.669375, rate:16000
    Audio duration: 4.0106875, rate:16000
    Audio duration: 3.413375, rate:16000
    Audio duration: 3.328, rate:16000
    Audio duration: 3.925375, rate:16000
    Audio duration: 2.645375, rate:16000
    finished


<h3>Plot 1 true en 1 false region data voor visualisatie</h3>
<p>Dus een plot van een label=1 en van een label=0</p>


```python
# Methode voor het plotten van de regio naar histogram voor visualisatie
def plotAudioRegion(signal, sample_rate, features, fignum, title, subplotNum):
    print('Frams [0] and filterbanks [1]' + str(features.shape))
    
    TimeSample = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(subplotNum+1)
    ax1.hist(signal)
    ax1.patch.set_facecolor('black')
    ax1.set_title(title + ': 10ms', fontsize=15)
    ax1.set_ylabel('Signal', fontsize=10)
    ax1.set_xlabel('Time', fontsize=10)
    
    T = getAudioDuration(signal, sample_rate)
    ax2 = fig.add_subplot(subplotNum+2)
    ax2.imshow(np.flipud(features.T), cmap=cm.jet, aspect='auto', extent=[0,T,0,13])
    ax2.set_title(title + ': 10ms', fontsize=15)
    ax2.set_ylabel('MFCC Coefficents', fontsize=10)
    ax2.set_xlabel('Time (s)', fontsize=10)
    
    plt.tight_layout()
```


```python
# Using Pandas for reading dataset csv
df_v2 = pd.read_csv(datasetDir + 'datasetboundary.csv', sep=',', skiprows=1, names=['region', 'label', 'sample_rate'])

# Get true labels only
trueData = df_v2.loc[df_v2.loc[:, 'label'] > 0, :]

# Get false labels only
falseData = df_v2.loc[df_v2.loc[:, 'label'] == 0, :]

print('Presentatie true regions data')
print(trueData.head())

print('Presentatie false regions data')
print(falseData.head())
```

    Presentatie true regions data
                                                   region  label  sample_rate
    0   0.03063869105700583|0.002808502583091155|0.013...      1        16000
    11  0.05230393235179385|-0.5528195766380677|-0.197...      1        16000
    22  -0.11285069440406623|-0.09468471219968748|0.04...      1        16000
    33  -0.0043863462056755505|-0.12700731775026952|-0...      1        16000
    44  0.7615775732452406|-0.37122520756508676|-2.532...      1        16000
    Presentatie false regions data
                                                  region  label  sample_rate
    1  0.11432089851650744|-0.84281116825059|-0.27725...      0        16000
    2  0.0864836929975997|-0.6890733413964334|-0.4354...      0        16000
    3  -0.035666205644276514|0.4795595844222028|0.456...      0        16000
    4  0.014038870992866848|-0.1368462476653683|-0.14...      0        16000
    5  -0.05846202225161079|-0.050493744002561944|-0....      0        16000


<p>Telling labels. Hier kun je zien hoeveel data van label 0 en 1 bestaan in de dataset.</p>


```python
import seaborn as sns

fig , ax = plt.subplots(figsize=(6,4))
sns.countplot(x='label', data=df_v2)
plt.title("Count of labels")
plt.show()
```


![png](output_17_0.png)



```python
# Transform to array from a stringdata and Get only 1 true region
regionSignalTrue = np.array([float(i) for i in trueData.loc[0]['region'].split('|')])
sample_rate = trueData.loc[0]['sample_rate']
print(len(regionSignalTrue))

# Transform to array from a stringdata and Get only 1 false region
regionSignalFalse = np.array([float(i) for i in falseData.loc[1]['region'].split('|')])

# Plot the true region data
plotAudioRegion(regionSignalTrue, sample_rate, transform2DFeatures(regionSignalTrue, 13), 1, 'True region', 220)

# Plot the false region data
plotAudioRegion(regionSignalFalse, sample_rate, transform2DFeatures(regionSignalFalse, 13), 3, 'False region', 222)

plt.show()
```

    130
    Frams [0] and filterbanks [1](10, 13)
    Frams [0] and filterbanks [1](10, 13)



![png](output_18_1.png)



![png](output_18_2.png)


<h1>Test of de gesneden regios geen leegte bevat</h1>
<p>De test realiseren we door de uitgesneden regios te concateneren tot 1 geheel audio. Ook kunnen we de volledige audio array afspelen om te analyseren.</p>


```python
datasetDir = '/datb/aphasia/languagedata/voxforge/dataset/'

folderpath = '/datb/aphasia/languagedata/voxforge/final/'

# Get all csv files
files = getFiles(folderpath)

regions = []

for x in range(0, len(files)-(len(files)-1), 1):

    filedict = readDict(files[x])
    audiopath = filedict[0]['audiopath']

#   Read audio
    sample_rate, audio = wav.read(audiopath)

    count = 0
    while count < len(filedict):
        # Get current element
        curr = filedict[count]

        # Get end-time and begin-time
        begin = float(curr['begin'])
        end = float(curr['end'])
        
        print('begin:{}, end:{}'.format(begin, end))

        # Get region frames
        region = audio[getTime(begin, sample_rate):getTime(end, sample_rate)]
        
        # Concatenate region to main region
        regions = np.concatenate((regions, region), axis=None)
        
        count += 1

print('finished')
```

    begin:0.5, end:0.9
    begin:0.9, end:1.3
    begin:1.3, end:1.4
    begin:1.4, end:1.5
    begin:1.5, end:1.7000000000000002
    begin:1.7000000000000002, end:1.85
    begin:1.85, end:2.15
    begin:2.15, end:2.3
    begin:2.45, end:2.7
    begin:2.7, end:2.9
    begin:2.9, end:3.45
    finished


<p>Hier kun je de audio array afspelen. De leegte bij het begin en het einde van de GEHELE audio is niet van belang want onze focus ligt op de interne van de gehele audio.</p>


```python
import IPython.display as ipd

ipd.Audio(regions, rate=16000)
```





                <audio controls="controls" >
                    <source src="data:audio/wav;base64,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QP9F/1v/UP9s/3H/Rf+d/3z/cf98/5j/uf+j/43/o/9x/5L/yf+z/6j/rv+u/8n/xP/E/53/mP/l/9r/vv+z/6j/yf8LAPv/vv/l/4f/6v/f//X/9f/U//D/IQD1/8//EAAWAGMAPAAFAAAATQBHAF0AMQBCAFgATQBNAEIAQgAhAE0AQgBHAIkAUgBHAIkAeQBCAH4A9f95AF0AiQBzAHkAYwCEAJoAuwCaAJQApQDRAJ8AuwDWALUAtQDRAAgB3ACwABMBxgAeAQgBAgH9AOwADQH3ABMB9wATAfIA9wDWANEAAgHhAOwA9wA5AcsAywDWAAIB5wDnANEAywC7AMYA7AACAdEAsgG7AOwAZQGGAQgBAgHcAMYAOQH9ADQBSgECAQIBwADyAPcAuwDAAMsAAgEIAf0AsACEAIQAuwDGAPIA4QCwAHkAywCwAIQAfgBzAKUAuwCfAG4AUgBuAHkApQCJAF0ALABSAIQAaABdAFgATQBdAEcAaAAWACwAIQAxAFIAPAAWADEAEABCAFgALAAQANT/8P8AACYAz//P/5L/vv8FAM//2v/l/7n/o/+5/43/uf/f/9T/qP+j/7P/rv+o/6j/d/+N/67/nf/f/6P/bP+o/2b/cf+o/4L/o/+H/2b/cf9m/y//QP9F/yr/Vv9F/0X/JP8q//j+bP9m/x//7f7+/g7/FP8q//7+1/4D/+j+Dv8Z/zr/FP8v/wn/JP81/yT/Cf8q/0X/Vv9F/xT/QP9L/4f/Ov9Q/y//d/9L/+r/bP9L/0X/4v5Q/9L+UP9m/xn/Kv8v/y//Nf/+/h//L/8f/0D/Yf8q/xn/JP9s/1v/Rf9b/2H//v5x/2H/cf9x/zr/Vv+j/43/W/9W/1D/Nf98/6P/h/+C/5j/h/+H/6P/kv+C/6j/kv/J/7n/rv+j/6P/rv/U/8T/h/+d/9r/1P/q/7n/h/+5/67/yf/1/+X/2v/7//v/6v/E/8//FgAWAPv/IQALACYAPABdAFIAPABCAEIAFgBNADwAIQBdAGMAtQCEADwAUgBSAGMAQgBdAFgAPABHAH4AfgCJAGgAiQC7AKUAmgDcALUAqgD3AMAAtQC1AAIBtQDRAPIA/QDhAPcANAEjAQ0BRAEuAQgBDQECAR4B4QAeAecAxgDAAMAAwADAALAA5wDGALUAwACaAI8AmgCqAJoAlACqAHkA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cwBjAFgAbgAWAFIAPABjAFIAXQBuAIQAcwCPAMAAwADAALAAsADWAKUA5wDhAP0A9wACAdwA8gD3AP0AywDcAB4B9wDyANEAywC7AAIB0QACAeEA3ADnAAIB/QDcAOwA5wApASkB1gDRANwAEwH3AOcAwADRANEAfgAYAf0AKQFgASkBEwG7AAIB/QATAf0ALgEeAdwACAEjAbAA7AD9ADQB0QC1AGMAeQDhANYA0QDWALUAxgC7ANEAiQCEAGgAcwCqALUAeQBSAIkAXQBHAIkANwAWAAsAaABHADwAAAALADcARwBjAEcAFgAsACEAWABSAFgATQBCAF0AeQCEAG4ARwA3AJoAQgA8ACEAQgAWAFgAGwCd//v/+//f/xsA1P/J/9//xP/l/8//3/+u/53/mP+u/43/cf/J/8n/s/+Y/4f/qP/a/67/uf+H/43/qP/U/7n/qP9x/4L/rv+j/3z/xP+o/+r/h/+N/2H/Ov8v/2z/nf9x/43/kv9h/zr/L/9m/1v/H/81/wn/S/9W/zX//v5F/x//QP9h/yr/Nf8q/2H/fP8f/0D/Gf9W/2z/bP9F/9f+8/7P/x//JP8U//j+7f7+/kD/Kv86/xT/1/7t/hn/+P5L/y//Gf8U/1D/Rf9L/zr/S/8v/1D/mP9m/1D/W/98/5L/z/++/53/kv+N/4L/uf+Y/3z/gv+z/6j/mP+z/3f/jf98/3f/o/+j/2b/h/+H/6P/jf9W/53/uf/J/8T/yf+d/6j/2v/P/9r/uf+o/6j/1P8bACwA+/8LABAAPAALAE0ABQAxADEAMQBYADwALAAbAEcAJgAFACwATQBNADwANwAxAAsATQBHAGgAaABHAEcAeQBoAHkAiQB+AJQAiQDRALUAhACUAJQAnwBuAMAAwADWANYA3ADnANEAwADRAOcA9wACASMBNAFEARMBKQEuAUQBSgFPAVoBRAE/ARMBLgEuARMBNAEpAQIBDQHRAOcA3AClAOwAtQCqAOr/0QATAbUACAEIARMBgQGwAP0A8gAIAf0AAgE0ASMBIwEjAWABDQHhAPIAwADRALUAlACPAHMATQBdAKUAlABuAGgAcwCJAGgAXQB+AIQAmgCEAJoAlABNAIQA5wCaAI8AbgCPAHkAhABoAE0ARwD7/yYAFgAFANr/8P/w//v/GwD7/ywAIQD1/wUAFgAQACEA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" type="audio/wav" />
                    Your browser does not support the audio element.
                </audio>
              



<h1>[Versie 3, Generator] - Een generator om ALLEEN de verschillen in de subregions op te slaan in de dataset.</h1>
<p>Dit doen we om te zien of de score beter wordt en om overfitting te vermijden.</p>

<h3>Benodigde methoden</h3>


```python
# Methode om het verschil van twee subregions terug te geven
def getDifferences(region, frame_size):
    sub_frame = int(frame_size/2)
    return region[0:sub_frame] - region[sub_frame:frame_size]


# Methode om de features van de verschil van de subregios uit de regios te kunnen krijgen.
def getRegionsFeaturesDifference(features_mfcc, side, boundary, frame_size, times):
    leftRegion = []
    rightRegion = []

    if 'L' in side:
        for walk in range(0, times):
            frame = boundary - (frame_size * walk)
            left = frame - frame_size
            right = frame
            differenceRegion = getDifferences(features_mfcc[left:right], frame_size)
            leftRegion.append(differenceRegion)

    if 'R' in side:
        for walk in range(0, times):
            frame = boundary + (frame_size * walk)
            left = frame
            right = frame + frame_size
            differenceRegion = getDifferences(features_mfcc[left:right], frame_size)
            rightRegion.append(differenceRegion)

    return leftRegion if 'L' in side else rightRegion
```

<h3>De generator</h3>


```python
datasetDir = '/datb/aphasia/languagedata/voxforge/dataset/'

folderpath = '/datb/aphasia/languagedata/voxforge/final/'

# Get all csv files
files = getFiles(folderpath)

multiply_ms = int(1000)
subRegion = int(10)
tsubRegion = int(subRegion / 2)
size_region = 5

# Save dataset in a csv file
with open(datasetDir + 'datasetboundary_difference.csv', 'w') as toWrite:

    fieldnames = ['region', 'label', 'sample_rate']
    writer = csv.DictWriter(toWrite, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, delimiter=',')

    writer.writeheader()

    for x in range(0, len(files)):

        filedict = readDict(files[x])
        audiopath = filedict[0]['audiopath']
        
#         Read audio
        sample_rate, audio = wav.read(audiopath)
        print('Audio duration: {}, rate:{}'.format(getAudioDuration(audio, sample_rate), sample_rate))
        
#         Transform audio to mfcc to get features
        features_mfcc = getSignalMFCC(audio, sample_rate)

        count = 1
        while count < len(filedict):
            # Get prev and current word element
            prevW = filedict[count - 1]
            currW = filedict[count]
            
            # Get prev end-time and current begin-time
            boundaryL = int(float(prevW['end']) * multiply_ms)
            boundaryR = int(float(currW['begin']) * multiply_ms)

#             # Get (true) left and right subregion frames
            tsubRegionL = features_mfcc[boundaryL-tsubRegion:boundaryL]
            tsubRegionR = features_mfcc[boundaryR:boundaryR + tsubRegion]

#             # Get difference of (false) subregions from left and right
            nRegionLfeatures = getRegionsFeaturesDifference(features_mfcc, 'L', boundaryL - tsubRegion, subRegion, size_region)
            nRegionRfeatures = getRegionsFeaturesDifference(features_mfcc, 'R', boundaryR + tsubRegion, subRegion, size_region)            
        
            # Difference (true) left subregion and right subregion to ONE True region
            tRegionFeatures = tsubRegionL - tsubRegionR
            
#           # Export to CSV
            exportDataCSV(tRegionFeatures, 1, sample_rate, writer)

            exportDatasCSV(nRegionLfeatures, 0, sample_rate, writer)
            exportDatasCSV(nRegionRfeatures, 0, sample_rate, writer)

            count += 1

print('finished')
```

    Audio duration: 3.7546875, rate:16000
    Audio duration: 2.7306875, rate:16000
    Audio duration: 3.4986875, rate:16000
    Audio duration: 2.7306875, rate:16000
    Audio duration: 3.669375, rate:16000
    Audio duration: 4.0106875, rate:16000
    Audio duration: 3.413375, rate:16000
    Audio duration: 3.328, rate:16000
    Audio duration: 3.925375, rate:16000
    Audio duration: 2.645375, rate:16000
    finished


<h4>Dataset na het uitvoeren van de "difference generator". file:"datasetboundary_difference.csv"</h4>


```python
# Using Pandas for reading dataset csv
df = pd.read_csv(datasetDir + 'datasetboundary_difference.csv', sep=',', skiprows=1, names=['region', 'label', 'sample_rate'])

print('Head:')
print(df.head())
print('----------------')
print('Tail:')
print(df.tail())
```

    Head:
                                                  region  label  sample_rate
    0  -0.005469213298211175|0.7779622038417742|0.677...      1        16000
    1  0.15707023369072176|-2.79486428991484|-1.73821...      0        16000
    2  -0.009104294089205933|-1.5565812334774602|-0.9...      0        16000
    3  -0.02562862411608719|1.7597042965140952|1.1330...      0        16000
    4  0.01737656392452429|2.224576509890566|-1.98389...      0        16000
    ----------------
    Tail:
                                                    region  label  sample_rate
    600  -0.1273367044986294|1.528383272412631|-0.74687...      0        16000
    601  -0.07745201518940412|-0.34143072185179413|1.89...      0        16000
    602  -0.014281864881123957|-1.5094740532703619|2.45...      0        16000
    603  -0.030374956625269078|-1.572207751864089|0.562...      0        16000
    604  0.09874311495447757|-0.7347260751012792|-1.307...      0        16000


<p>Telling labels. Hier kun je zien hoeveel data van label 0 en 1 bestaan in de dataset.</p>


```python
import seaborn as sns

fig , ax = plt.subplots(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Count of labels")
plt.show()
```


![png](output_31_0.png)



```python

```


```python

```


```python

```


```python

```
