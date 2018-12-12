
<h1>Scraper for scraping data from fon.hum.uva.nl</h1>
<p>(c) Koray</p>
<p>Deze scraper (een soort BOT) haalt de gewenste data zoals audio en daaraan gerelateerde gesproken tekst af van de website fon.hum.uva.nl. Deze data is de ruwe data en kan gebruikt worden voor toekomstige aanmaak van datasets voor het trainen van modellen zoals neurale netwerk. Deze data wordt vervolgens gebruikt bij een aligner om de audio te knippen op zin niveau want de audio bestanden bestaan uit zinnen i.p.v. een zin. Aligner behoort tot het onderdeel "Data Preperation".</p>

<h2>Script voor het downloaden van de gewenste data.</h2>
<p>Deze script doet zich voor als een normale website bezoeker en download alle data op de website in één keer.</p>


```python
# Needed packages
import os, requests
from subprocess import call
from bs4 import BeautifulSoup

#Init urls
audioUrl = 'http://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFAcorpus/SLspeech/chunks/'
textUrl = 'http://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFAcorpus/SLcorpus/Transcriptions/'
headers = {'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"}

# Get the request body with HEADERS
audiofolderRequest = requests.get(url=audioUrl, headers=headers)
textfolderRequest = requests.get(url=textUrl, headers=headers)

# Init BeautifulSoup with HTML parser
audiosoup = BeautifulSoup(audiofolderRequest.text, 'html.parser')
textsoup = BeautifulSoup(textfolderRequest.text, 'html.parser')

# Init folder lists
audiofolderlinks = []
textfolderlinks = []

# Add all links of the folder in de lists
for x in audiosoup.find_all(name='a', href=True):
    if x.get_text() == x.attrs['href']:
        audiofolderlinks.append(audioUrl + x.get_text())

for x in textsoup.find_all(name='a', href=True):
    if x.get_text() == x.attrs['href']:
        textfolderlinks.append(textUrl + x.get_text())

# Init links list for audio and text
audiolinks = []
textlinks = []

# Loop through folder lists and request with headers to add the audio links to the lists
for folderlink in audiofolderlinks:
    audioRequest = requests.get(url=folderlink, headers=headers)
    foldersoup = BeautifulSoup(audioRequest.text, 'html.parser')
    for audiolink in foldersoup.find_all(name='a', href=True):
        if audiolink.get_text() == audiolink.attrs['href']:
            if audiolink.get_text().split('.')[-1] != 'txt':
                audiolinks.append(folderlink+audiolink.get_text())

# Loop through folder lists and request with headers to add the text links to the lists
for folderlink in textfolderlinks:
    textRequest = requests.get(url=folderlink, headers=headers)
    foldersoup = BeautifulSoup(textRequest.text, 'html.parser')
    for textlink in foldersoup.find_all(name='a', href=True):
        if textlink.get_text() == textlink.attrs['href']:
            textlinks.append(folderlink+textlink.get_text())

# Loop through audio list and curl the audio link and download it on the server
for audiolink in audiolinks:
    finalFilename = '/datb/aphasia/dutchaudio/originalUva/' + audiolink.split('/')[-1]
    curlAudio = 'curl -0 ' + audiolink + ' -o ' + finalFilename
    call(curlAudio, shell = True)
    print('Finished: {}'.format(audiolink.split('/')[-1]))

# Loop through text list and curl the text link and download it on the server
for textlink in textlinks:
    finalFilename = '/datb/aphasia/dutchaudio/originalUva/' + textlink.split('/')[-1]
    curlText = 'curl -0 ' + textlink + ' -o ' + finalFilename
    call(curlText, shell = True)
    print('Finished: {}'.format(textlink.split('/')[-1]))
```

    Finished: F20N1FPA1.aifc
    Finished: F20N1FPA2.aifc
    Finished: F20N1FR1.aifc
    Finished: F20N1FR2.aifc
    Finished: F20N1FS.aifc
    Finished: F20N1FT1.aifc
    Finished: F20N1FT2.aifc
    Finished: F20N1FT3.aifc
    Finished: F20N1FT4.aifc
    Finished: F20N1FT5.aifc
    Finished: F20N1FT6.aifc
    Finished: F20N1FT7.aifc
    Finished: F20N1FT8.aifc
    Finished: F20N1FT9.aifc
    Finished: F20N1FT10.aifc
    Finished: F20N1FT11.aifc
    Finished: F20N1FW.aifc
    Finished: F20N1FY.aifc
    Finished: F20N1G1N1.aifc
    Finished: F20N1G1N2.aifc
    Finished: F20N1G1T1.aifc
    Finished: F20N1G1T2.aifc
    Finished: F20N1G2N1.aifc
    Finished: F20N1G2N2.aifc
    Finished: F20N1G2T1.aifc
    Finished: F20N1G2T2.aifc
    Finished: F20N1VI1.aifc
    Finished: F20N1VI2.aifc
    Finished: F20N1VI3.aifc
    Finished: F20N1VI4.aifc
    Finished: F20N1VI5.aifc
    Finished: F20N1VI6.aifc
    Finished: F20N1VI7.aifc
    Finished: F20N1VI8.aifc
    Finished: F20N1VI9.aifc
    Finished: F20N1VI10.aifc
    Finished: F20N1VI11.aifc
    Finished: F20N1VI12.aifc
    Finished: F20N1VI13.aifc
    Finished: F20N1VI14.aifc
    Finished: F20N1VI15.aifc
    Finished: F20N1VI16.aifc
    Finished: F20N1VI17.aifc
    Finished: F20N1VI18.aifc
    Finished: F20N1VI19.aifc
    Finished: F20N1VI20.aifc
    Finished: F20N1VI21.aifc
    Finished: F20N1VI22.aifc
    Finished: F20N1VI23.aifc
    Finished: F20N1VI24.aifc
    Finished: F20N1VI25.aifc
    Finished: F20N1VI26.aifc
    Finished: F20N2FPB1.aifc
    Finished: F20N2FPB2.aifc
    Finished: F20N2G1N1.aifc
    Finished: F20N2G1N2.aifc
    Finished: F20N2G1T1.aifc
    Finished: F20N2G1T2.aifc
    Finished: F20N2PS.aifc
    Finished: F20N2VR1.aifc
    Finished: F20N2VS.aifc
    Finished: F20N2VT1.aifc
    Finished: F20N2VT2.aifc
    Finished: F20N2VT3.aifc
    Finished: F20N2VT4.aifc
    Finished: F20N2VT5.aifc
    Finished: F20N2VT6.aifc
    Finished: F20N2VT7.aifc
    Finished: F20N2VT8.aifc
    Finished: F20N2VT9.aifc
    Finished: F20N2VT10.aifc
    Finished: F20N2VT11.aifc
    Finished: F20N2VT12.aifc
    Finished: F20N2VT13.aifc
    Finished: F20N2VT14.aifc
    Finished: F20N2VT15.aifc
    Finished: F20N2VT16.aifc
    Finished: F20N2VT17.aifc
    Finished: F20N2VT18.aifc
    Finished: F20N2VT19.aifc
    Finished: F20N2VT20.aifc
    Finished: F20N2VT21.aifc
    Finished: F20N2VT22.aifc
    Finished: F20N2VT23.aifc
    Finished: F20N2VT24.aifc
    Finished: F20N2VT25.aifc
    Finished: F20N2VT26.aifc
    Finished: F20N2VW.aifc
    Finished: F20N2VY.aifc
    Finished: F24I1FPA1.aifc
    Finished: F24I1FPA2.aifc
    Finished: F24I1FR1.aifc
    Finished: F24I1FR2.aifc
    Finished: F24I1FS.aifc
    Finished: F24I1FT1.aifc
    Finished: F24I1FT2.aifc
    Finished: F24I1FT3.aifc
    Finished: F24I1FT4.aifc
    Finished: F24I1FT5.aifc
    Finished: F24I1FT6.aifc
    Finished: F24I1FT7.aifc
    Finished: F24I1FT8.aifc
    Finished: F24I1FT9.aifc
    Finished: F24I1FT10.aifc
    Finished: F24I1FT11.aifc
    Finished: F24I1FW.aifc
    Finished: F24I1FY.aifc
    Finished: F24I1G1N1.aifc
    Finished: F24I1G1N2.aifc
    Finished: F24I1G1T1.aifc
    Finished: F24I1G1T2.aifc
    Finished: F24I1G2N1.aifc
    Finished: F24I1G2N2.aifc
    Finished: F24I1G2T1.aifc
    Finished: F24I1G2T2.aifc
    Finished: F24I1VI1.aifc
    Finished: F24I1VI2.aifc
    Finished: F24I1VI3.aifc
    Finished: F24I1VI4.aifc
    Finished: F24I1VI5.aifc
    Finished: F24I1VI6.aifc
    Finished: F24I1VI7.aifc
    Finished: F24I1VI8.aifc
    Finished: F24I1VI9.aifc
    Finished: F24I1VI10.aifc
    Finished: F24I1VI11.aifc
    Finished: F24I1VI12.aifc
    Finished: F24I1VI13.aifc
    Finished: F24I1VI14.aifc
    Finished: F24I1VI15.aifc
    Finished: F24I1VI16.aifc
    Finished: F24I1VI17.aifc
    Finished: F24I1VI18.aifc
    Finished: F24I1VI19.aifc
    Finished: F24I1VI20.aifc
    Finished: F24I1VI21.aifc
    Finished: F24I1VI22.aifc
    Finished: F24I1VI23.aifc
    Finished: F24I1VI24.aifc
    Finished: F24I2FPB1.aifc
    Finished: F24I2FPB2.aifc
    Finished: F24I2G1N1.aifc
    Finished: F24I2G1N2.aifc
    Finished: F24I2G1T1.aifc
    Finished: F24I2G1T2.aifc
    Finished: F24I2PS.aifc
    Finished: F24I2VR1.aifc
    Finished: F24I2VS.aifc
    Finished: F24I2VT1.aifc
    Finished: F24I2VT2.aifc
    Finished: F24I2VT3.aifc
    Finished: F24I2VT4.aifc
    Finished: F24I2VT5.aifc
    Finished: F24I2VT6.aifc
    Finished: F24I2VT7.aifc
    Finished: F24I2VT8.aifc
    Finished: F24I2VT9.aifc
    Finished: F24I2VT10.aifc
    Finished: F24I2VT11.aifc
    Finished: F24I2VT12.aifc
    Finished: F24I2VT13.aifc
    Finished: F24I2VT14.aifc
    Finished: F24I2VT15.aifc
    Finished: F24I2VT16.aifc
    Finished: F24I2VT17.aifc
    Finished: F24I2VT18.aifc
    Finished: F24I2VT19.aifc
    Finished: F24I2VT20.aifc
    Finished: F24I2VT21.aifc
    Finished: F24I2VT22.aifc
    Finished: F24I2VT23.aifc
    Finished: F24I2VT24.aifc
    Finished: F24I2VW.aifc
    Finished: F24I2VY.aifc
    Finished: F28G1FPA1.aifc
    Finished: F28G1FPA2.aifc
    Finished: F28G1FR1.aifc
    Finished: F28G1FR2.aifc
    Finished: F28G1FS.aifc
    Finished: F28G1FT1.aifc
    Finished: F28G1FT2.aifc
    Finished: F28G1FT3.aifc
    Finished: F28G1FT4.aifc
    Finished: F28G1FT5.aifc
    Finished: F28G1FT6.aifc
    Finished: F28G1FT7.aifc
    Finished: F28G1FT8.aifc
    Finished: F28G1FT9.aifc
    Finished: F28G1FT10.aifc
    Finished: F28G1FT11.aifc
    Finished: F28G1FW.aifc
    Finished: F28G1FY.aifc
    Finished: F28G1G1N1.aifc
    Finished: F28G1G1N2.aifc
    Finished: F28G1G1T1.aifc
    Finished: F28G1G1T2.aifc
    Finished: F28G1G2N1.aifc
    Finished: F28G1G2N2.aifc
    Finished: F28G1G2T1.aifc
    Finished: F28G1G2T2.aifc
    Finished: F28G1VI1.aifc
    Finished: F28G1VI2.aifc
    Finished: F28G1VI3.aifc
    Finished: F28G1VI4.aifc
    Finished: F28G1VI5.aifc
    Finished: F28G1VI6.aifc
    Finished: F28G1VI7.aifc
    Finished: F28G1VI8.aifc
    Finished: F28G1VI9.aifc
    Finished: F28G1VI10.aifc
    Finished: F28G1VI11.aifc
    Finished: F28G2FPB1.aifc
    Finished: F28G2FPB2.aifc
    Finished: F28G2G1N1.aifc
    Finished: F28G2G1N2.aifc
    Finished: F28G2G1T1.aifc
    Finished: F28G2G1T2.aifc
    Finished: F28G2PS.aifc
    Finished: F28G2VR1.aifc
    Finished: F28G2VS.aifc
    Finished: F28G2VT1.aifc
    Finished: F28G2VT2.aifc
    Finished: F28G2VT3.aifc
    Finished: F28G2VT4.aifc
    Finished: F28G2VT5.aifc
    Finished: F28G2VT6.aifc
    Finished: F28G2VT7.aifc
    Finished: F28G2VT8.aifc
    Finished: F28G2VT9.aifc
    Finished: F28G2VT10.aifc
    Finished: F28G2VT11.aifc
    Finished: F28G2VW.aifc
    Finished: F28G2VY.aifc
    Finished: F40L1FPA1.aifc
    Finished: F40L1FPA2.aifc
    Finished: F40L1FR1.aifc
    Finished: F40L1FR2.aifc
    Finished: F40L1FS.aifc
    Finished: F40L1FT1.aifc
    Finished: F40L1FT2.aifc
    Finished: F40L1FT3.aifc
    Finished: F40L1FT4.aifc
    Finished: F40L1FT5.aifc
    Finished: F40L1FT6.aifc
    Finished: F40L1FT7.aifc
    Finished: F40L1FT8.aifc
    Finished: F40L1FT9.aifc
    Finished: F40L1FT10.aifc
    Finished: F40L1FT11.aifc
    Finished: F40L1FW.aifc
    Finished: F40L1FY.aifc
    Finished: F40L1G1N1.aifc
    Finished: F40L1G1N2.aifc
    Finished: F40L1G1T1.aifc
    Finished: F40L1G1T2.aifc
    Finished: F40L1G2N1.aifc
    Finished: F40L1G2N2.aifc
    Finished: F40L1G2T1.aifc
    Finished: F40L1G2T2.aifc
    Finished: F40L1VI1.aifc
    Finished: F40L1VI2.aifc
    Finished: F40L1VI3.aifc
    Finished: F40L1VI4.aifc
    Finished: F40L1VI5.aifc
    Finished: F40L1VI6.aifc
    Finished: F40L1VI7.aifc
    Finished: F40L1VI8.aifc
    Finished: F40L1VI9.aifc
    Finished: F40L2FPB1.aifc
    Finished: F40L2FPB2.aifc
    Finished: F40L2G1N1.aifc
    Finished: F40L2G1N2.aifc
    Finished: F40L2G1T1.aifc
    Finished: F40L2G1T2.aifc
    Finished: F40L2PS.aifc
    Finished: F40L2VR1.aifc
    Finished: F40L2VS.aifc
    Finished: F40L2VT1.aifc
    Finished: F40L2VT2.aifc
    Finished: F40L2VT3.aifc
    Finished: F40L2VT4.aifc
    Finished: F40L2VT5.aifc
    Finished: F40L2VT6.aifc
    Finished: F40L2VT7.aifc
    Finished: F40L2VT8.aifc
    Finished: F40L2VT9.aifc
    Finished: F40L2VW.aifc
    Finished: F40L2VY.aifc
    Finished: F60E1FPA1.aifc
    Finished: F60E1FPA2.aifc
    Finished: F60E1FR1.aifc
    Finished: F60E1FR2.aifc
    Finished: F60E1FS.aifc
    Finished: F60E1FT1.aifc
    Finished: F60E1FT2.aifc
    Finished: F60E1FT3.aifc
    Finished: F60E1FT4.aifc
    Finished: F60E1FT5.aifc
    Finished: F60E1FT6.aifc
    Finished: F60E1FT7.aifc
    Finished: F60E1FT8.aifc
    Finished: F60E1FT9.aifc
    Finished: F60E1FT10.aifc
    Finished: F60E1FT11.aifc
    Finished: F60E1FW.aifc
    Finished: F60E1FY.aifc
    Finished: F60E1G1N1.aifc
    Finished: F60E1G1N2.aifc
    Finished: F60E1G1T1.aifc
    Finished: F60E1G1T2.aifc
    Finished: F60E1G2N1.aifc
    Finished: F60E1G2N2.aifc
    Finished: F60E1G2T1.aifc
    Finished: F60E1G2T2.aifc
    Finished: F60E1VI1.aifc
    Finished: F60E1VI2.aifc
    Finished: F60E1VI3.aifc
    Finished: F60E1VI4.aifc
    Finished: F60E1VI5.aifc
    Finished: F60E1VI6.aifc
    Finished: F60E1VI7.aifc
    Finished: F60E1VI8.aifc
    Finished: F60E1VI9.aifc
    Finished: F60E1VI10.aifc
    Finished: F60E1VI11.aifc
    Finished: F60E1VI12.aifc
    Finished: F60E1VI13.aifc
    Finished: F60E1VI14.aifc
    Finished: F60E1VI15.aifc
    Finished: F60E2FPB1.aifc
    Finished: F60E2FPB2.aifc
    Finished: F60E2G1N1.aifc
    Finished: F60E2G1N2.aifc
    Finished: F60E2G1T1.aifc
    Finished: F60E2G1T2.aifc
    Finished: F60E2PS.aifc
    Finished: F60E2VR1.aifc
    Finished: F60E2VS.aifc
    Finished: F60E2VT1.aifc
    Finished: F60E2VT2.aifc
    Finished: F60E2VT3.aifc
    Finished: F60E2VT4.aifc
    Finished: F60E2VT5.aifc
    Finished: F60E2VT6.aifc
    Finished: F60E2VT7.aifc
    Finished: F60E2VT8.aifc
    Finished: F60E2VT9.aifc
    Finished: F60E2VT10.aifc
    Finished: F60E2VT11.aifc
    Finished: F60E2VT12.aifc
    Finished: F60E2VT13.aifc
    Finished: F60E2VT14.aifc
    Finished: F60E2VT15.aifc
    Finished: F60E2VW.aifc
    Finished: F60E2VY.aifc
    Finished: M15R1FPA1.aifc
    Finished: M15R1FPA2.aifc
    Finished: M15R1FR1.aifc
    Finished: M15R1FR2.aifc
    Finished: M15R1FS.aifc
    Finished: M15R1FT1.aifc
    Finished: M15R1FT2.aifc
    Finished: M15R1FT3.aifc
    Finished: M15R1FT4.aifc
    Finished: M15R1FT5.aifc
    Finished: M15R1FT6.aifc
    Finished: M15R1FT7.aifc
    Finished: M15R1FT8.aifc
    Finished: M15R1FT9.aifc
    Finished: M15R1FT10.aifc
    Finished: M15R1FT11.aifc
    Finished: M15R1FW.aifc
    Finished: M15R1FY.aifc
    Finished: M15R1G1N1.aifc
    Finished: M15R1G1N2.aifc
    Finished: M15R1G1T1.aifc
    Finished: M15R1G1T2.aifc
    Finished: M15R1G2N1.aifc
    Finished: M15R1G2N2.aifc
    Finished: M15R1G2T1.aifc
    Finished: M15R1G2T2.aifc
    Finished: M15R1VI1.aifc
    Finished: M15R1VI2.aifc
    Finished: M15R1VI3.aifc
    Finished: M15R1VI4.aifc
    Finished: M15R1VI5.aifc
    Finished: M15R1VI6.aifc
    Finished: M15R1VI7.aifc
    Finished: M15R1VI8.aifc
    Finished: M15R1VI9.aifc
    Finished: M15R1VI10.aifc
    Finished: M15R1VI11.aifc
    Finished: M15R1VI12.aifc
    Finished: M15R1VI13.aifc
    Finished: M15R1VI14.aifc
    Finished: M15R1VI15.aifc
    Finished: M15R1VI16.aifc
    Finished: M15R1VI17.aifc
    Finished: M15R1VI18.aifc
    Finished: M15R2FPB1.aifc
    Finished: M15R2FPB2.aifc
    Finished: M15R2G1N1.aifc
    Finished: M15R2G1N2.aifc
    Finished: M15R2G1T1.aifc
    Finished: M15R2G1T2.aifc
    Finished: M15R2PS.aifc
    Finished: M15R2VR1.aifc
    Finished: M15R2VS.aifc
    Finished: M15R2VT1.aifc
    Finished: M15R2VT2.aifc
    Finished: M15R2VT3.aifc
    Finished: M15R2VT4.aifc
    Finished: M15R2VT5.aifc
    Finished: M15R2VT6.aifc
    Finished: M15R2VT7.aifc
    Finished: M15R2VT8.aifc
    Finished: M15R2VT9.aifc
    Finished: M15R2VT10.aifc
    Finished: M15R2VT11.aifc
    Finished: M15R2VT12.aifc
    Finished: M15R2VT13.aifc
    Finished: M15R2VT14.aifc
    Finished: M15R2VT15.aifc
    Finished: M15R2VT16.aifc
    Finished: M15R2VT17.aifc
    Finished: M15R2VT18.aifc
    Finished: M15R2VW.aifc
    Finished: M15R2VY.aifc
    Finished: M40K1FPA1.aifc
    Finished: M40K1FPA2.aifc
    Finished: M40K1FR1.aifc
    Finished: M40K1FR2.aifc
    Finished: M40K1FS.aifc
    Finished: M40K1FT1.aifc
    Finished: M40K1FT2.aifc
    Finished: M40K1FT3.aifc
    Finished: M40K1FT4.aifc
    Finished: M40K1FT5.aifc
    Finished: M40K1FT6.aifc
    Finished: M40K1FT7.aifc
    Finished: M40K1FT8.aifc
    Finished: M40K1FT9.aifc
    Finished: M40K1FT10.aifc
    Finished: M40K1FT11.aifc
    Finished: M40K1FW.aifc
    Finished: M40K1FY.aifc
    Finished: M40K1G1N1.aifc
    Finished: M40K1G1N2.aifc
    Finished: M40K1G1T1.aifc
    Finished: M40K1G1T2.aifc
    Finished: M40K1G2N1.aifc
    Finished: M40K1G2N2.aifc
    Finished: M40K1G2T1.aifc
    Finished: M40K1G2T2.aifc
    Finished: M40K1VI1.aifc
    Finished: M40K1VI2.aifc
    Finished: M40K1VI3.aifc
    Finished: M40K1VI4.aifc
    Finished: M40K1VI5.aifc
    Finished: M40K1VI6.aifc
    Finished: M40K1VI7.aifc
    Finished: M40K1VI8.aifc
    Finished: M40K1VI9.aifc
    Finished: M40K1VI10.aifc
    Finished: M40K1VI11.aifc
    Finished: M40K1VI12.aifc
    Finished: M40K2FPB1.aifc
    Finished: M40K2FPB2.aifc
    Finished: M40K2G1N1.aifc
    Finished: M40K2G1N2.aifc
    Finished: M40K2G1T1.aifc
    Finished: M40K2G1T2.aifc
    Finished: M40K2PS.aifc
    Finished: M40K2VR1.aifc
    Finished: M40K2VS.aifc
    Finished: M40K2VT1.aifc
    Finished: M40K2VT2.aifc
    Finished: M40K2VT3.aifc
    Finished: M40K2VT4.aifc
    Finished: M40K2VT5.aifc
    Finished: M40K2VT6.aifc
    Finished: M40K2VT7.aifc
    Finished: M40K2VT8.aifc
    Finished: M40K2VT9.aifc
    Finished: M40K2VT10.aifc
    Finished: M40K2VT11.aifc
    Finished: M40K2VT12.aifc
    Finished: M40K2VW.aifc
    Finished: M40K2VY.aifc
    Finished: M56H1FPA1.aifc
    Finished: M56H1FPA2.aifc
    Finished: M56H1FR1.aifc
    Finished: M56H1FR2.aifc
    Finished: M56H1FS.aifc
    Finished: M56H1FT1.aifc
    Finished: M56H1FT2.aifc
    Finished: M56H1FT3-1.aifc
    Finished: M56H1FT3.aifc
    Finished: M56H1FT4.aifc
    Finished: M56H1FT5.aifc
    Finished: M56H1FT6.aifc
    Finished: M56H1FT7.aifc
    Finished: M56H1FT8.aifc
    Finished: M56H1FT9.aifc
    Finished: M56H1FT10.aifc
    Finished: M56H1FT11.aifc
    Finished: M56H1FW.aifc
    Finished: M56H1FY.aifc
    Finished: M56H1G1.aifc
    Finished: M56H1G1N1.aifc
    Finished: M56H1G1N2.aifc
    Finished: M56H1G1T1.aifc
    Finished: M56H1G1T2.aifc
    Finished: M56H1G2N1.aifc
    Finished: M56H1G2N2.aifc
    Finished: M56H1G2T1.aifc
    Finished: M56H1G2T2.aifc
    Finished: M56H1VI1.aifc
    Finished: M56H1VI2.aifc
    Finished: M56H1VI3.aifc
    Finished: M56H1VI4.aifc
    Finished: M56H1VI5.aifc
    Finished: M56H1VI6.aifc
    Finished: M56H1VI7.aifc
    Finished: M56H1VI8.aifc
    Finished: M56H1VI9.aifc
    Finished: M56H1VI10.aifc
    Finished: M56H1VI11.aifc
    Finished: M56H1VI12.aifc
    Finished: M56H2FPB1.aifc
    Finished: M56H2FPB2.aifc
    Finished: M56H2G1N1.aifc
    Finished: M56H2G1N2.aifc
    Finished: M56H2G1T1.aifc
    Finished: M56H2G1T2.aifc
    Finished: M56H2PS.aifc
    Finished: M56H2VR1.aifc
    Finished: M56H2VS.aifc
    Finished: M56H2VT1.aifc
    Finished: M56H2VT2.aifc
    Finished: M56H2VT3.aifc
    Finished: M56H2VT4.aifc
    Finished: M56H2VT5.aifc
    Finished: M56H2VT6.aifc
    Finished: M56H2VT7.aifc
    Finished: M56H2VT8.aifc
    Finished: M56H2VT9.aifc
    Finished: M56H2VT10.aifc
    Finished: M56H2VT11.aifc
    Finished: M56H2VT12.aifc
    Finished: M56H2VW.aifc
    Finished: M56H2VY.aifc
    Finished: M58D1FPA1.aifc
    Finished: M58D1FPA2.aifc
    Finished: M58D1FR1.aifc
    Finished: M58D1FR2.aifc
    Finished: M58D1FS.aifc
    Finished: M58D1FT1.aifc
    Finished: M58D1FT2.aifc
    Finished: M58D1FT3.aifc
    Finished: M58D1FT4.aifc
    Finished: M58D1FT5.aifc
    Finished: M58D1FT6.aifc
    Finished: M58D1FT7.aifc
    Finished: M58D1FT8.aifc
    Finished: M58D1FT9.aifc
    Finished: M58D1FT10.aifc
    Finished: M58D1FT11.aifc
    Finished: M58D1FW.aifc
    Finished: M58D1FY.aifc
    Finished: M58D1G1N1.aifc
    Finished: M58D1G1T1.aifc
    Finished: M58D1G1T2.aifc
    Finished: M58D1G2N1.aifc
    Finished: M58D1G2N2.aifc
    Finished: M58D1G2T1.aifc
    Finished: M58D1G2T2.aifc
    Finished: M58D1VI1.aifc
    Finished: M58D1VI2.aifc
    Finished: M58D1VI3.aifc
    Finished: M58D1VI4.aifc
    Finished: M58D1VI5.aifc
    Finished: M58D1VI6.aifc
    Finished: M58D1VI7.aifc
    Finished: M58D1VI8.aifc
    Finished: M58D1VI9.aifc
    Finished: M58D1VI10.aifc
    Finished: M58D2FPB1.aifc
    Finished: M58D2FPB2.aifc
    Finished: M58D2G1N1.aifc
    Finished: M58D2G1N2.aifc
    Finished: M58D2G1T1.aifc
    Finished: M58D2G1T2.aifc
    Finished: M58D2PS.aifc
    Finished: M58D2VR1.aifc
    Finished: M58D2VS.aifc
    Finished: M58D2VT1.aifc
    Finished: M58D2VT2.aifc
    Finished: M58D2VT3.aifc
    Finished: M58D2VT4.aifc
    Finished: M58D2VT5.aifc
    Finished: M58D2VT6.aifc
    Finished: M58D2VT7.aifc
    Finished: M58D2VT8.aifc
    Finished: M58D2VT9.aifc
    Finished: M58D2VT10.aifc
    Finished: M58D2VW.aifc
    Finished: M58D2VY.aifc
    Finished: M66O1FPA1.aifc
    Finished: M66O1FPA2.aifc
    Finished: M66O1FR1.aifc
    Finished: M66O1FR2.aifc
    Finished: M66O1FS.aifc
    Finished: M66O1FT1.aifc
    Finished: M66O1FT2.aifc
    Finished: M66O1FT3.aifc
    Finished: M66O1FT4.aifc
    Finished: M66O1FT5.aifc
    Finished: M66O1FT6.aifc
    Finished: M66O1FT7.aifc
    Finished: M66O1FT8.aifc
    Finished: M66O1FT9.aifc
    Finished: M66O1FT10.aifc
    Finished: M66O1FT11.aifc
    Finished: M66O1FW.aifc
    Finished: M66O1FY.aifc
    Finished: M66O1G1N1.aifc
    Finished: M66O1G1N2.aifc
    Finished: M66O1G1T1.aifc
    Finished: M66O1G1T2.aifc
    Finished: M66O1G2N1.aifc
    Finished: M66O1G2N2.aifc
    Finished: M66O1G2T1.aifc
    Finished: M66O1G2T2.aifc
    Finished: M66O1VI1.aifc
    Finished: M66O1VI2.aifc
    Finished: M66O1VI3.aifc
    Finished: M66O1VI4.aifc
    Finished: M66O1VI5.aifc
    Finished: M66O1VI6.aifc
    Finished: M66O1VI7.aifc
    Finished: M66O1VI8.aifc
    Finished: M66O1VI9.aifc
    Finished: M66O1VI10.aifc
    Finished: M66O1VI11.aifc
    Finished: M66O1VI12.aifc
    Finished: M66O1VI13.aifc
    Finished: M66O1VI14.aifc
    Finished: M66O1VI15.aifc
    Finished: M66O1VI16.aifc
    Finished: M66O1VI17.aifc
    Finished: M66O1VI18.aifc
    Finished: M66O2FPB1.aifc
    Finished: M66O2FPB2.aifc
    Finished: M66O2G1N1.aifc
    Finished: M66O2G1N2.aifc
    Finished: M66O2G1T1.aifc
    Finished: M66O2G1T2.aifc
    Finished: M66O2PS.aifc
    Finished: M66O2VR1.aifc
    Finished: M66O2VS.aifc
    Finished: M66O2VT1.aifc
    Finished: M66O2VT2.aifc
    Finished: M66O2VT3.aifc
    Finished: M66O2VT4.aifc
    Finished: M66O2VT5.aifc
    Finished: M66O2VT6.aifc
    Finished: M66O2VT7.aifc
    Finished: M66O2VT8.aifc
    Finished: M66O2VT9.aifc
    Finished: M66O2VT10.aifc
    Finished: M66O2VT11.aifc
    Finished: M66O2VT12.aifc
    Finished: M66O2VT13.aifc
    Finished: M66O2VT14.aifc
    Finished: M66O2VT15.aifc
    Finished: M66O2VT16.aifc
    Finished: M66O2VT17.aifc
    Finished: M66O2VT18.aifc
    Finished: M66O2VW.aifc
    Finished: M66O2VY.aifc
    Finished: F20N1FPA1.txt
    Finished: F20N1FPA2.txt
    Finished: F20N1FR1.txt
    Finished: F20N1FR2.txt
    Finished: F20N1FS.txt
    Finished: F20N1FT1.txt
    Finished: F20N1FT2.txt
    Finished: F20N1FT3.txt
    Finished: F20N1FT4.txt
    Finished: F20N1FT5.txt
    Finished: F20N1FT6.txt
    Finished: F20N1FT7.txt
    Finished: F20N1FT8.txt
    Finished: F20N1FT9.txt
    Finished: F20N1FT10.txt
    Finished: F20N1FT11.txt
    Finished: F20N1FW.txt
    Finished: F20N1FY.txt
    Finished: F20N1VI1.txt
    Finished: F20N1VI2.txt
    Finished: F20N1VI3.txt
    Finished: F20N1VI4.txt
    Finished: F20N1VI5.txt
    Finished: F20N1VI6.txt
    Finished: F20N1VI7.txt
    Finished: F20N1VI8.txt
    Finished: F20N1VI9.txt
    Finished: F20N1VI10.txt
    Finished: F20N1VI11.txt
    Finished: F20N1VI12.txt
    Finished: F20N1VI13.txt
    Finished: F20N1VI14.txt
    Finished: F20N1VI15.txt
    Finished: F20N1VI16.txt
    Finished: F20N1VI17.txt
    Finished: F20N1VI18.txt
    Finished: F20N1VI19.txt
    Finished: F20N1VI20.txt
    Finished: F20N1VI21.txt
    Finished: F20N1VI22.txt
    Finished: F20N1VI23.txt
    Finished: F20N1VI24.txt
    Finished: F20N1VI25.txt
    Finished: F20N1VI26.txt
    Finished: F20N2FPB1.txt
    Finished: F20N2FPB2.txt
    Finished: F20N2PS.txt
    Finished: F20N2VR1.txt
    Finished: F20N2VS.txt
    Finished: F20N2VT1.txt
    Finished: F20N2VT2.txt
    Finished: F20N2VT3.txt
    Finished: F20N2VT4.txt
    Finished: F20N2VT5.txt
    Finished: F20N2VT6.txt
    Finished: F20N2VT7.txt
    Finished: F20N2VT8.txt
    Finished: F20N2VT9.txt
    Finished: F20N2VT10.txt
    Finished: F20N2VT11.txt
    Finished: F20N2VT12.txt
    Finished: F20N2VT13.txt
    Finished: F20N2VT14.txt
    Finished: F20N2VT15.txt
    Finished: F20N2VT16.txt
    Finished: F20N2VT17.txt
    Finished: F20N2VT18.txt
    Finished: F20N2VT19.txt
    Finished: F20N2VT20.txt
    Finished: F20N2VT21.txt
    Finished: F20N2VT22.txt
    Finished: F20N2VT23.txt
    Finished: F20N2VT24.txt
    Finished: F20N2VT25.txt
    Finished: F20N2VT26.txt
    Finished: F20N2VW.txt
    Finished: F20N2VY.txt
    Finished: F24I1FPA1.txt
    Finished: F24I1FPA2.txt
    Finished: F24I1FS.txt
    Finished: F24I1FT1.txt
    Finished: F24I1FT2.txt
    Finished: F24I1FT3.txt
    Finished: F24I1FT4.txt
    Finished: F24I1FT5.txt
    Finished: F24I1FT6.txt
    Finished: F24I1FT7.txt
    Finished: F24I1FT8.txt
    Finished: F24I1FT9.txt
    Finished: F24I1FT10.txt
    Finished: F24I1FT11.txt
    Finished: F24I1FW.txt
    Finished: F24I1FY.txt
    Finished: F24I1VI1.txt
    Finished: F24I1VI2.txt
    Finished: F24I1VI3.txt
    Finished: F24I1VI4.txt
    Finished: F24I1VI5.txt
    Finished: F24I1VI6.txt
    Finished: F24I1VI7.txt
    Finished: F24I1VI8.txt
    Finished: F24I1VI9.txt
    Finished: F24I1VI10.txt
    Finished: F24I1VI11.txt
    Finished: F24I1VI12.txt
    Finished: F24I1VI13.txt
    Finished: F24I1VI14.txt
    Finished: F24I1VI15.txt
    Finished: F24I1VI16.txt
    Finished: F24I1VI17.txt
    Finished: F24I1VI18.txt
    Finished: F24I1VI19.txt
    Finished: F24I1VI20.txt
    Finished: F24I1VI21.txt
    Finished: F24I1VI22.txt
    Finished: F24I1VI23.txt
    Finished: F24I1VI24.txt
    Finished: F24I2FPB1.txt
    Finished: F24I2FPB2.txt
    Finished: F24I2PS.txt
    Finished: F24I2VS.txt
    Finished: F24I2VT1.txt
    Finished: F24I2VT2.txt
    Finished: F24I2VT3.txt
    Finished: F24I2VT4.txt
    Finished: F24I2VT5.txt
    Finished: F24I2VT6.txt
    Finished: F24I2VT7.txt
    Finished: F24I2VT8.txt
    Finished: F24I2VT9.txt
    Finished: F24I2VT10.txt
    Finished: F24I2VT11.txt
    Finished: F24I2VT12.txt
    Finished: F24I2VT13.txt
    Finished: F24I2VT14.txt
    Finished: F24I2VT15.txt
    Finished: F24I2VT16.txt
    Finished: F24I2VT17.txt
    Finished: F24I2VT18.txt
    Finished: F24I2VT19.txt
    Finished: F24I2VT20.txt
    Finished: F24I2VT21.txt
    Finished: F24I2VT22.txt
    Finished: F24I2VT23.txt
    Finished: F24I2VT24.txt
    Finished: F24I2VW.txt
    Finished: F24I2VY.txt
    Finished: F26A1FPA1.txt
    Finished: F26A1FPA2.txt
    Finished: F26A1FS.txt
    Finished: F26A1FT1.txt
    Finished: F26A1FT2.txt
    Finished: F26A1FT3.txt
    Finished: F26A1FT4.txt
    Finished: F26A1FT5.txt
    Finished: F26A1FT6.txt
    Finished: F26A1FT7.txt
    Finished: F26A1FT8.txt
    Finished: F26A1FT9.txt
    Finished: F26A1FT10.txt
    Finished: F26A1FT11.txt
    Finished: F26A1FW.txt
    Finished: F26A1FY.txt
    Finished: F26A1VI1.txt
    Finished: F26A1VI2.txt
    Finished: F26A1VI3.txt
    Finished: F26A1VI4.txt
    Finished: F26A1VI5.txt
    Finished: F26A1VI6.txt
    Finished: F26A2FPB1.txt
    Finished: F26A2FPB2.txt
    Finished: F26A2PS.txt
    Finished: F26A2VS.txt
    Finished: F26A2VT1.txt
    Finished: F26A2VT2.txt
    Finished: F26A2VT3.txt
    Finished: F26A2VT4.txt
    Finished: F26A2VT5.txt
    Finished: F26A2VT6.txt
    Finished: F26A2VW.txt
    Finished: F26A2VY.txt
    Finished: F27B1FPA1.txt
    Finished: F27B1FPA2.txt
    Finished: F27B1FS.txt
    Finished: F27B1FT1.txt
    Finished: F27B1FT2.txt
    Finished: F27B1FT3.txt
    Finished: F27B1FT4.txt
    Finished: F27B1FT5.txt
    Finished: F27B1FT6.txt
    Finished: F27B1FT7.txt
    Finished: F27B1FT8.txt
    Finished: F27B1FT9.txt
    Finished: F27B1FT10.txt
    Finished: F27B1FT11.txt
    Finished: F27B1FW.txt
    Finished: F27B1FY.txt
    Finished: F27B1VI1.txt
    Finished: F27B1VI2.txt
    Finished: F27B1VI3.txt
    Finished: F27B1VI4.txt
    Finished: F27B1VI5.txt
    Finished: F27B1VI6.txt
    Finished: F27B1VI7.txt
    Finished: F27B1VI8.txt
    Finished: F27B1VI9.txt
    Finished: F27B1VI10.txt
    Finished: F27B1VI11.txt
    Finished: F27B1VI12.txt
    Finished: F27B2FPB1.txt
    Finished: F27B2FPB2.txt
    Finished: F27B2PS.txt
    Finished: F27B2VS.txt
    Finished: F27B2VT1.txt
    Finished: F27B2VT2.txt
    Finished: F27B2VT3.txt
    Finished: F27B2VT4.txt
    Finished: F27B2VT5.txt
    Finished: F27B2VT6.txt
    Finished: F27B2VT7.txt
    Finished: F27B2VT8.txt
    Finished: F27B2VT9.txt
    Finished: F27B2VT10.txt
    Finished: F27B2VT11.txt
    Finished: F27B2VT12.txt
    Finished: F27B2VW.txt
    Finished: F27B2VY.txt
    Finished: F28G1FPA1.txt
    Finished: F28G1FPA2.txt
    Finished: F28G1FR1.txt
    Finished: F28G1FR2.txt
    Finished: F28G1FS.txt
    Finished: F28G1FT1.txt
    Finished: F28G1FT2.txt
    Finished: F28G1FT3.txt
    Finished: F28G1FT4.txt
    Finished: F28G1FT5.txt
    Finished: F28G1FT6.txt
    Finished: F28G1FT7.txt
    Finished: F28G1FT8.txt
    Finished: F28G1FT9.txt
    Finished: F28G1FT10.txt
    Finished: F28G1FT11.txt
    Finished: F28G1FW.txt
    Finished: F28G1FY.txt
    Finished: F28G1VI1.txt
    Finished: F28G1VI2.txt
    Finished: F28G1VI3.txt
    Finished: F28G1VI4.txt
    Finished: F28G1VI5.txt
    Finished: F28G1VI6.txt
    Finished: F28G1VI7.txt
    Finished: F28G1VI8.txt
    Finished: F28G1VI9.txt
    Finished: F28G1VI10.txt
    Finished: F28G1VI11.txt
    Finished: F28G2FPB1.txt
    Finished: F28G2FPB2.txt
    Finished: F28G2PS.txt
    Finished: F28G2VR1.txt
    Finished: F28G2VS.txt
    Finished: F28G2VT1.txt
    Finished: F28G2VT2.txt
    Finished: F28G2VT3.txt
    Finished: F28G2VT4.txt
    Finished: F28G2VT5.txt
    Finished: F28G2VT6.txt
    Finished: F28G2VT7.txt
    Finished: F28G2VT8.txt
    Finished: F28G2VT9.txt
    Finished: F28G2VT10.txt
    Finished: F28G2VT11.txt
    Finished: F28G2VW.txt
    Finished: F28G2VY.txt
    Finished: F29J1FPA1.txt
    Finished: F29J1FPA2.txt
    Finished: F29J1FS.txt
    Finished: F29J1FT1.txt
    Finished: F29J1FT2.txt
    Finished: F29J1FT3.txt
    Finished: F29J1FT4.txt
    Finished: F29J1FT5.txt
    Finished: F29J1FT6.txt
    Finished: F29J1FT7.txt
    Finished: F29J1FT8.txt
    Finished: F29J1FT9.txt
    Finished: F29J1FT10.txt
    Finished: F29J1FT11.txt
    Finished: F29J1FW.txt
    Finished: F29J1FY.txt
    Finished: F29J1VI1.txt
    Finished: F29J1VI2.txt
    Finished: F29J1VI3.txt
    Finished: F29J1VI4.txt
    Finished: F29J1VI5.txt
    Finished: F29J1VI6.txt
    Finished: F29J1VI7.txt
    Finished: F29J1VI8.txt
    Finished: F29J1VI9.txt
    Finished: F29J1VI10.txt
    Finished: F29J1VI11.txt
    Finished: F29J1VI12.txt
    Finished: F29J1VI13.txt
    Finished: F29J1VI14.txt
    Finished: F29J1VI15.txt
    Finished: F29J1VI16.txt
    Finished: F29J1VI17.txt
    Finished: F29J1VI18.txt
    Finished: F29J2FPB1.txt
    Finished: F29J2FPB2.txt
    Finished: F29J2PS.txt
    Finished: F29J2VS.txt
    Finished: F29J2VT1.txt
    Finished: F29J2VT2.txt
    Finished: F29J2VT3.txt
    Finished: F29J2VT4.txt
    Finished: F29J2VT5.txt
    Finished: F29J2VT6.txt
    Finished: F29J2VT7.txt
    Finished: F29J2VT8.txt
    Finished: F29J2VT9.txt
    Finished: F29J2VT10.txt
    Finished: F29J2VT11.txt
    Finished: F29J2VT12.txt
    Finished: F29J2VT13.txt
    Finished: F29J2VT14.txt
    Finished: F29J2VT15.txt
    Finished: F29J2VT16.txt
    Finished: F29J2VT17.txt
    Finished: F29J2VT18.txt
    Finished: F29J2VW.txt
    Finished: F29J2VY.txt
    Finished: F40L1FPA1.txt
    Finished: F40L1FPA2.txt
    Finished: F40L1FR1.txt
    Finished: F40L1FR2.txt
    Finished: F40L1FS.txt
    Finished: F40L1FT1.txt
    Finished: F40L1FT2.txt
    Finished: F40L1FT3.txt
    Finished: F40L1FT4.txt
    Finished: F40L1FT5.txt
    Finished: F40L1FT6.txt
    Finished: F40L1FT7.txt
    Finished: F40L1FT8.txt
    Finished: F40L1FT9.txt
    Finished: F40L1FT10.txt
    Finished: F40L1FT11.txt
    Finished: F40L1FW.txt
    Finished: F40L1FY.txt
    Finished: F40L1VI1.txt
    Finished: F40L1VI2.txt
    Finished: F40L1VI3.txt
    Finished: F40L1VI4.txt
    Finished: F40L1VI5.txt
    Finished: F40L1VI6.txt
    Finished: F40L1VI7.txt
    Finished: F40L1VI8.txt
    Finished: F40L1VI9.txt
    Finished: F40L2FPB1.txt
    Finished: F40L2FPB2.txt
    Finished: F40L2PS.txt
    Finished: F40L2VR1.txt
    Finished: F40L2VS.txt
    Finished: F40L2VT1.txt
    Finished: F40L2VT2.txt
    Finished: F40L2VT3.txt
    Finished: F40L2VT4.txt
    Finished: F40L2VT5.txt
    Finished: F40L2VT6.txt
    Finished: F40L2VT7.txt
    Finished: F40L2VT8.txt
    Finished: F40L2VT9.txt
    Finished: F40L2VW.txt
    Finished: F40L2VY.txt
    Finished: F56F1FPA1.txt
    Finished: F56F1FPA2.txt
    Finished: F56F1FS.txt
    Finished: F56F1FT1.txt
    Finished: F56F1FT2.txt
    Finished: F56F1FT3.txt
    Finished: F56F1FT4.txt
    Finished: F56F1FT5.txt
    Finished: F56F1FT6.txt
    Finished: F56F1FT7.txt
    Finished: F56F1FT8.txt
    Finished: F56F1FT9.txt
    Finished: F56F1FT10.txt
    Finished: F56F1FT11.txt
    Finished: F56F1FW.txt
    Finished: F56F1FY.txt
    Finished: F56F1VI1.txt
    Finished: F56F1VI2.txt
    Finished: F56F1VI3.txt
    Finished: F56F1VI4.txt
    Finished: F56F1VI5.txt
    Finished: F56F1VI6.txt
    Finished: F56F1VI7.txt
    Finished: F56F1VI8.txt
    Finished: F56F1VI9.txt
    Finished: F56F1VI10.txt
    Finished: F56F1VI11.txt
    Finished: F56F1VI12.txt
    Finished: F56F1VI13.txt
    Finished: F56F2FPB1.txt
    Finished: F56F2FPB2.txt
    Finished: F56F2PS.txt
    Finished: F56F2VS.txt
    Finished: F56F2VT1.txt
    Finished: F56F2VT2.txt
    Finished: F56F2VT3.txt
    Finished: F56F2VT4.txt
    Finished: F56F2VT5.txt
    Finished: F56F2VT6.txt
    Finished: F56F2VT7.txt
    Finished: F56F2VT8.txt
    Finished: F56F2VT9.txt
    Finished: F56F2VT10.txt
    Finished: F56F2VT11.txt
    Finished: F56F2VT12.txt
    Finished: F56F2VT13.txt
    Finished: F56F2VW.txt
    Finished: F56F2VY.txt
    Finished: F60E1FPA1.txt
    Finished: F60E1FPA2.txt
    Finished: F60E1FR1.txt
    Finished: F60E1FR2.txt
    Finished: F60E1FS.txt
    Finished: F60E1FT1.txt
    Finished: F60E1FT2.txt
    Finished: F60E1FT3.txt
    Finished: F60E1FT4.txt
    Finished: F60E1FT5.txt
    Finished: F60E1FT6.txt
    Finished: F60E1FT7.txt
    Finished: F60E1FT8.txt
    Finished: F60E1FT9.txt
    Finished: F60E1FT10.txt
    Finished: F60E1FT11.txt
    Finished: F60E1FW.txt
    Finished: F60E1FY.txt
    Finished: F60E1VI1.txt
    Finished: F60E1VI2.txt
    Finished: F60E1VI3.txt
    Finished: F60E1VI4.txt
    Finished: F60E1VI5.txt
    Finished: F60E1VI6.txt
    Finished: F60E1VI7.txt
    Finished: F60E1VI8.txt
    Finished: F60E1VI9.txt
    Finished: F60E1VI10.txt
    Finished: F60E1VI11.txt
    Finished: F60E1VI12.txt
    Finished: F60E1VI13.txt
    Finished: F60E1VI14.txt
    Finished: F60E1VI15.txt
    Finished: F60E2FPB1.txt
    Finished: F60E2FPB2.txt
    Finished: F60E2PS.txt
    Finished: F60E2VR1.txt
    Finished: F60E2VS.txt
    Finished: F60E2VT1.txt
    Finished: F60E2VT2.txt
    Finished: F60E2VT3.txt
    Finished: F60E2VT4.txt
    Finished: F60E2VT5.txt
    Finished: F60E2VT6.txt
    Finished: F60E2VT7.txt
    Finished: F60E2VT8.txt
    Finished: F60E2VT9.txt
    Finished: F60E2VT10.txt
    Finished: F60E2VT11.txt
    Finished: F60E2VT12.txt
    Finished: F60E2VT13.txt
    Finished: F60E2VT14.txt
    Finished: F60E2VT15.txt
    Finished: F60E2VW.txt
    Finished: F60E2VY.txt
    Finished: M15R1FPA1.txt
    Finished: M15R1FPA2.txt
    Finished: M15R1FR1.txt
    Finished: M15R1FR2.txt
    Finished: M15R1FS.txt
    Finished: M15R1FT1.txt
    Finished: M15R1FT2.txt
    Finished: M15R1FT3.txt
    Finished: M15R1FT4.txt
    Finished: M15R1FT5.txt
    Finished: M15R1FT6.txt
    Finished: M15R1FT7.txt
    Finished: M15R1FT8.txt
    Finished: M15R1FT9.txt
    Finished: M15R1FT10.txt
    Finished: M15R1FT11.txt
    Finished: M15R1FW.txt
    Finished: M15R1FY.txt
    Finished: M15R1VI1.txt
    Finished: M15R1VI2.txt
    Finished: M15R1VI3.txt
    Finished: M15R1VI4.txt
    Finished: M15R1VI5.txt
    Finished: M15R1VI6.txt
    Finished: M15R1VI7.txt
    Finished: M15R1VI8.txt
    Finished: M15R1VI9.txt
    Finished: M15R1VI10.txt
    Finished: M15R1VI11.txt
    Finished: M15R1VI12.txt
    Finished: M15R1VI13.txt
    Finished: M15R1VI14.txt
    Finished: M15R1VI15.txt
    Finished: M15R1VI16.txt
    Finished: M15R1VI17.txt
    Finished: M15R1VI18.txt
    Finished: M15R2FPB1.txt
    Finished: M15R2FPB2.txt
    Finished: M15R2PS.txt
    Finished: M15R2VR1.txt
    Finished: M15R2VS.txt
    Finished: M15R2VT1.txt
    Finished: M15R2VT2.txt
    Finished: M15R2VT3.txt
    Finished: M15R2VT4.txt
    Finished: M15R2VT5.txt
    Finished: M15R2VT6.txt
    Finished: M15R2VT7.txt
    Finished: M15R2VT8.txt
    Finished: M15R2VT9.txt
    Finished: M15R2VT10.txt
    Finished: M15R2VT11.txt
    Finished: M15R2VT12.txt
    Finished: M15R2VT13.txt
    Finished: M15R2VT14.txt
    Finished: M15R2VT15.txt
    Finished: M15R2VT16.txt
    Finished: M15R2VT17.txt
    Finished: M15R2VT18.txt
    Finished: M15R2VW.txt
    Finished: M15R2VY.txt
    Finished: M23Q1FPA1.txt
    Finished: M23Q1FPA2.txt
    Finished: M23Q1FS.txt
    Finished: M23Q1FT1.txt
    Finished: M23Q1FT2.txt
    Finished: M23Q1FT3.txt
    Finished: M23Q1FT4.txt
    Finished: M23Q1FT5.txt
    Finished: M23Q1FT6.txt
    Finished: M23Q1FT7.txt
    Finished: M23Q1FT8.txt
    Finished: M23Q1FT9.txt
    Finished: M23Q1FT10.txt
    Finished: M23Q1FT11.txt
    Finished: M23Q1FW.txt
    Finished: M23Q1FY.txt
    Finished: M23Q1VI1.txt
    Finished: M23Q1VI2.txt
    Finished: M23Q1VI3.txt
    Finished: M23Q1VI4.txt
    Finished: M23Q1VI5.txt
    Finished: M23Q1VI6.txt
    Finished: M23Q1VI7.txt
    Finished: M23Q1VI8.txt
    Finished: M23Q1VI9.txt
    Finished: M23Q1VI10.txt
    Finished: M23Q1VI11.txt
    Finished: M23Q1VI12.txt
    Finished: M23Q1VI13.txt
    Finished: M23Q1VI14.txt
    Finished: M23Q1VI15.txt
    Finished: M23Q1VI16.txt
    Finished: M23Q1VI17.txt
    Finished: M23Q1VI18.txt
    Finished: M23Q1VI19.txt
    Finished: M23Q1VI20.txt
    Finished: M23Q1VI21.txt
    Finished: M23Q2FPB1.txt
    Finished: M23Q2FPB2.txt
    Finished: M23Q2PS.txt
    Finished: M23Q2VS.txt
    Finished: M23Q2VT1.txt
    Finished: M23Q2VT2.txt
    Finished: M23Q2VT3.txt
    Finished: M23Q2VT4.txt
    Finished: M23Q2VT5.txt
    Finished: M23Q2VT6.txt
    Finished: M23Q2VT7.txt
    Finished: M23Q2VT8.txt
    Finished: M23Q2VT9.txt
    Finished: M23Q2VT10.txt
    Finished: M23Q2VT11.txt
    Finished: M23Q2VT12.txt
    Finished: M23Q2VT13.txt
    Finished: M23Q2VT14.txt
    Finished: M23Q2VT15.txt
    Finished: M23Q2VT16.txt
    Finished: M23Q2VT17.txt
    Finished: M23Q2VT18.txt
    Finished: M23Q2VT19.txt
    Finished: M23Q2VT20.txt
    Finished: M23Q2VT21.txt
    Finished: M23Q2VW.txt
    Finished: M23Q2VY.txt
    Finished: M40K1FPA1.txt
    Finished: M40K1FPA2.txt
    Finished: M40K1FR1.txt
    Finished: M40K1FR2.txt
    Finished: M40K1FS.txt
    Finished: M40K1FT1.txt
    Finished: M40K1FT2.txt
    Finished: M40K1FT3.txt
    Finished: M40K1FT4.txt
    Finished: M40K1FT5.txt
    Finished: M40K1FT6.txt
    Finished: M40K1FT7.txt
    Finished: M40K1FT8.txt
    Finished: M40K1FT9.txt
    Finished: M40K1FT10.txt
    Finished: M40K1FT11.txt
    Finished: M40K1FW.txt
    Finished: M40K1FY.txt
    Finished: M40K1VI1.txt
    Finished: M40K1VI2.txt
    Finished: M40K1VI3.txt
    Finished: M40K1VI4.txt
    Finished: M40K1VI5.txt
    Finished: M40K1VI6.txt
    Finished: M40K1VI7.txt
    Finished: M40K1VI8.txt
    Finished: M40K1VI9.txt
    Finished: M40K1VI10.txt
    Finished: M40K1VI11.txt
    Finished: M40K1VI12.txt
    Finished: M40K2FPB1.txt
    Finished: M40K2FPB2.txt
    Finished: M40K2PS.txt
    Finished: M40K2VR1.txt
    Finished: M40K2VS.txt
    Finished: M40K2VT1.txt
    Finished: M40K2VT2.txt
    Finished: M40K2VT3.txt
    Finished: M40K2VT4.txt
    Finished: M40K2VT5.txt
    Finished: M40K2VT6.txt
    Finished: M40K2VT7.txt
    Finished: M40K2VT8.txt
    Finished: M40K2VT9.txt
    Finished: M40K2VT10.txt
    Finished: M40K2VT11.txt
    Finished: M40K2VT12.txt
    Finished: M40K2VW.txt
    Finished: M40K2VY.txt
    Finished: M46C1FPA1.txt
    Finished: M46C1FPA2.txt
    Finished: M46C1FS.txt
    Finished: M46C1FT1.txt
    Finished: M46C1FT2.txt
    Finished: M46C1FT3.txt
    Finished: M46C1FT4.txt
    Finished: M46C1FT5.txt
    Finished: M46C1FT6.txt
    Finished: M46C1FT7.txt
    Finished: M46C1FT8.txt
    Finished: M46C1FT9.txt
    Finished: M46C1FT10.txt
    Finished: M46C1FT11.txt
    Finished: M46C1FW.txt
    Finished: M46C1FY.txt
    Finished: M46C1VI1.txt
    Finished: M46C1VI2.txt
    Finished: M46C1VI3.txt
    Finished: M46C1VI4.txt
    Finished: M46C1VI5.txt
    Finished: M46C1VI6.txt
    Finished: M46C1VI7.txt
    Finished: M46C1VI8.txt
    Finished: M46C1VI9.txt
    Finished: M46C1VI10.txt
    Finished: M46C1VI11.txt
    Finished: M46C1VI12.txt
    Finished: M46C2FPB1.txt
    Finished: M46C2FPB2.txt
    Finished: M46C2PS.txt
    Finished: M46C2VS.txt
    Finished: M46C2VT1.txt
    Finished: M46C2VT2.txt
    Finished: M46C2VT3.txt
    Finished: M46C2VT4.txt
    Finished: M46C2VT5.txt
    Finished: M46C2VT6.txt
    Finished: M46C2VT7.txt
    Finished: M46C2VT8.txt
    Finished: M46C2VT9.txt
    Finished: M46C2VT10.txt
    Finished: M46C2VT11.txt
    Finished: M46C2VT12.txt
    Finished: M46C2VW.txt
    Finished: M46C2VY.txt
    Finished: M56H1FPA1.txt
    Finished: M56H1FPA2.txt
    Finished: M56H1FR1.txt
    Finished: M56H1FR2.txt
    Finished: M56H1FS.txt
    Finished: M56H1FT1.txt
    Finished: M56H1FT2.txt
    Finished: M56H1FT3.txt
    Finished: M56H1FT4.txt
    Finished: M56H1FT5.txt
    Finished: M56H1FT6.txt
    Finished: M56H1FT7.txt
    Finished: M56H1FT8.txt
    Finished: M56H1FT9.txt
    Finished: M56H1FT10.txt
    Finished: M56H1FT11.txt
    Finished: M56H1FW.txt
    Finished: M56H1FY.txt
    Finished: M56H1VI1.txt
    Finished: M56H1VI2.txt
    Finished: M56H1VI3.txt
    Finished: M56H1VI4.txt
    Finished: M56H1VI5.txt
    Finished: M56H1VI6.txt
    Finished: M56H1VI7.txt
    Finished: M56H1VI8.txt
    Finished: M56H1VI9.txt
    Finished: M56H1VI10.txt
    Finished: M56H1VI11.txt
    Finished: M56H1VI12.txt
    Finished: M56H2FPB1.txt
    Finished: M56H2FPB2.txt
    Finished: M56H2PS.txt
    Finished: M56H2VR1.txt
    Finished: M56H2VS.txt
    Finished: M56H2VT1.txt
    Finished: M56H2VT2.txt
    Finished: M56H2VT3.txt
    Finished: M56H2VT4.txt
    Finished: M56H2VT5.txt
    Finished: M56H2VT6.txt
    Finished: M56H2VT7.txt
    Finished: M56H2VT8.txt
    Finished: M56H2VT9.txt
    Finished: M56H2VT10.txt
    Finished: M56H2VT11.txt
    Finished: M56H2VT12.txt
    Finished: M56H2VW.txt
    Finished: M56H2VY.txt
    Finished: M56M1FPA1.txt
    Finished: M56M1FPA2.txt
    Finished: M56M1FS.txt
    Finished: M56M1FT1.txt
    Finished: M56M1FT2.txt
    Finished: M56M1FT3.txt
    Finished: M56M1FT4.txt
    Finished: M56M1FT5.txt
    Finished: M56M1FT6.txt
    Finished: M56M1FT7.txt
    Finished: M56M1FT8.txt
    Finished: M56M1FT9.txt
    Finished: M56M1FT10.txt
    Finished: M56M1FT11.txt
    Finished: M56M1FW.txt
    Finished: M56M1FY.txt
    Finished: M56M1VI1.txt
    Finished: M56M1VI2.txt
    Finished: M56M1VI3.txt
    Finished: M56M1VI4.txt
    Finished: M56M1VI5.txt
    Finished: M56M1VI6.txt
    Finished: M56M1VI7.txt
    Finished: M56M1VI8.txt
    Finished: M56M1VI9.txt
    Finished: M56M1VI10.txt
    Finished: M56M1VI11.txt
    Finished: M56M1VI12.txt
    Finished: M56M1VI13.txt
    Finished: M56M1VI14.txt
    Finished: M56M1VI15.txt
    Finished: M56M1VI16.txt
    Finished: M56M1VI17.txt
    Finished: M56M1VI18.txt
    Finished: M56M1VI19.txt
    Finished: M56M1VI20.txt
    Finished: M56M1VI21.txt
    Finished: M56M1VI22.txt
    Finished: M56M1VI23.txt
    Finished: M56M1VI24.txt
    Finished: M56M1VI25.txt
    Finished: M56M1VI26.txt
    Finished: M56M1VI27.txt
    Finished: M56M1VI28.txt
    Finished: M56M1VI29.txt
    Finished: M56M1VI30.txt
    Finished: M56M1VI31.txt
    Finished: M56M1VI32.txt
    Finished: M56M1VI33.txt
    Finished: M56M1VI34.txt
    Finished: M56M1VI35.txt
    Finished: M56M1VI36.txt
    Finished: M56M1VI37.txt
    Finished: M56M1VI38.txt
    Finished: M56M1VI39.txt
    Finished: M56M1VI40.txt
    Finished: M56M1VI41.txt
    Finished: M56M2FPB1.txt
    Finished: M56M2FPB2.txt
    Finished: M56M2PS.txt
    Finished: M56M2VS.txt
    Finished: M56M2VT1.txt
    Finished: M56M2VT2.txt
    Finished: M56M2VT3.txt
    Finished: M56M2VT4.txt
    Finished: M56M2VT5.txt
    Finished: M56M2VT6.txt
    Finished: M56M2VT7.txt
    Finished: M56M2VT8.txt
    Finished: M56M2VT9.txt
    Finished: M56M2VT10.txt
    Finished: M56M2VT11.txt
    Finished: M56M2VT12.txt
    Finished: M56M2VT13.txt
    Finished: M56M2VT14.txt
    Finished: M56M2VT15.txt
    Finished: M56M2VT16.txt
    Finished: M56M2VT17.txt
    Finished: M56M2VT18.txt
    Finished: M56M2VT19.txt
    Finished: M56M2VT20.txt
    Finished: M56M2VT21.txt
    Finished: M56M2VT22.txt
    Finished: M56M2VT23.txt
    Finished: M56M2VT24.txt
    Finished: M56M2VT25.txt
    Finished: M56M2VT26.txt
    Finished: M56M2VT27.txt
    Finished: M56M2VT28.txt
    Finished: M56M2VT29.txt
    Finished: M56M2VT30.txt
    Finished: M56M2VT31.txt
    Finished: M56M2VT32.txt
    Finished: M56M2VT33.txt
    Finished: M56M2VT34.txt
    Finished: M56M2VT35.txt
    Finished: M56M2VT36.txt
    Finished: M56M2VT37.txt
    Finished: M56M2VT38.txt
    Finished: M56M2VT39.txt
    Finished: M56M2VT40.txt
    Finished: M56M2VT41.txt
    Finished: M56M2VW.txt
    Finished: M56M2VY.txt
    Finished: M58D1FPA1.txt
    Finished: M58D1FPA2.txt
    Finished: M58D1FS.txt
    Finished: M58D1FT1.txt
    Finished: M58D1FT2.txt
    Finished: M58D1FT3.txt
    Finished: M58D1FT4.txt
    Finished: M58D1FT5.txt
    Finished: M58D1FT6.txt
    Finished: M58D1FT7.txt
    Finished: M58D1FT8.txt
    Finished: M58D1FT9.txt
    Finished: M58D1FT10.txt
    Finished: M58D1FT11.txt
    Finished: M58D1FW.txt
    Finished: M58D1FY.txt
    Finished: M58D1VI1.txt
    Finished: M58D1VI2.txt
    Finished: M58D1VI3.txt
    Finished: M58D1VI4.txt
    Finished: M58D1VI5.txt
    Finished: M58D1VI6.txt
    Finished: M58D1VI7.txt
    Finished: M58D1VI8.txt
    Finished: M58D1VI9.txt
    Finished: M58D1VI10.txt
    Finished: M58D2FPB1.txt
    Finished: M58D2FPB2.txt
    Finished: M58D2PS.txt
    Finished: M58D2VS.txt
    Finished: M58D2VT1.txt
    Finished: M58D2VT2.txt
    Finished: M58D2VT3.txt
    Finished: M58D2VT4.txt
    Finished: M58D2VT5.txt
    Finished: M58D2VT6.txt
    Finished: M58D2VT7.txt
    Finished: M58D2VT8.txt
    Finished: M58D2VT9.txt
    Finished: M58D2VT10.txt
    Finished: M58D2VW.txt
    Finished: M58D2VY.txt
    Finished: M66O1FPA1.txt
    Finished: M66O1FPA2.txt
    Finished: M66O1FR1.txt
    Finished: M66O1FR2.txt
    Finished: M66O1FS.txt
    Finished: M66O1FT1.txt
    Finished: M66O1FT2.txt
    Finished: M66O1FT3.txt
    Finished: M66O1FT4.txt
    Finished: M66O1FT5.txt
    Finished: M66O1FT6.txt
    Finished: M66O1FT7.txt
    Finished: M66O1FT8.txt
    Finished: M66O1FT9.txt
    Finished: M66O1FT10.txt
    Finished: M66O1FT11.txt
    Finished: M66O1FW.txt
    Finished: M66O1FY.txt
    Finished: M66O1VI1.txt
    Finished: M66O1VI2.txt
    Finished: M66O1VI3.txt
    Finished: M66O1VI4.txt
    Finished: M66O1VI5.txt
    Finished: M66O1VI6.txt
    Finished: M66O1VI7.txt
    Finished: M66O1VI8.txt
    Finished: M66O1VI9.txt
    Finished: M66O1VI10.txt
    Finished: M66O1VI11.txt
    Finished: M66O1VI12.txt
    Finished: M66O1VI13.txt
    Finished: M66O1VI14.txt
    Finished: M66O1VI15.txt
    Finished: M66O1VI16.txt
    Finished: M66O1VI17.txt
    Finished: M66O1VI18.txt
    Finished: M66O2FPB1.txt
    Finished: M66O2FPB2.txt
    Finished: M66O2PS.txt
    Finished: M66O2VR1.txt
    Finished: M66O2VS.txt
    Finished: M66O2VT1.txt
    Finished: M66O2VT2.txt
    Finished: M66O2VT3.txt
    Finished: M66O2VT4.txt
    Finished: M66O2VT5.txt
    Finished: M66O2VT6.txt
    Finished: M66O2VT7.txt
    Finished: M66O2VT8.txt
    Finished: M66O2VT9.txt
    Finished: M66O2VT10.txt
    Finished: M66O2VT11.txt
    Finished: M66O2VT12.txt
    Finished: M66O2VT13.txt
    Finished: M66O2VT14.txt
    Finished: M66O2VT15.txt
    Finished: M66O2VT16.txt
    Finished: M66O2VT17.txt
    Finished: M66O2VT18.txt
    Finished: M66O2VW.txt
    Finished: M66O2VY.txt



```python
print('audio package items: {}'.format(len(audiolinks)))
print('text package items: {}'.format(len(textlinks)))
```

    audio package items: 681
    text package items: 954


<h2>Test</h2>
<p>This is for checking if there are files that are not related to any audio files. This is a important part for alignment process of sentences and words.</p>
<p>Als er een tekst of een audio bestaat die niet gerelateerd is aan een tekst of audio dan heeft die data geen toegevoegde waarde.</p>


```python
import glob

directoryPath = '/datb/aphasia/dutchaudio/originalUva/'

audiodict = {}
textdict = {}

def initAllfiles():
    audiotmp = glob.glob(directoryPath+'*.aifc')
    texttmp = glob.glob(directoryPath+'*.txt')
    
    for file in audiotmp:
        audio = file.split('/')[-1]
        audiodict[audio] = file
        
    for file in texttmp:
        text = file.split('/')[-1]
        textdict[text] = file
    pass


initAllfiles()

count = 0
notIn = 0

for file in textdict:
    fileTmp = file.split('.')[0]+'.aifc'
    if fileTmp in audiodict:
        count += 1
    else:
        notIn += 1
        print('Not in audiodict: {}'.format(file))
        print(textdict[file])

print('In: {}'.format(count))
print('Not in: {}'.format(notIn))
```

    Not in audiodict: F26A1FW.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FW.txt
    Not in audiodict: F27B1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI5.txt
    Not in audiodict: F26A1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT9.txt
    Not in audiodict: F29J1VI16.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI16.txt
    Not in audiodict: M46C1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI2.txt
    Not in audiodict: F29J1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT8.txt
    Not in audiodict: F56F1FY.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FY.txt
    Not in audiodict: M23Q2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT11.txt
    Not in audiodict: F27B1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT7.txt
    Not in audiodict: F56F2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2FPB2.txt
    Not in audiodict: F29J1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI2.txt
    Not in audiodict: M46C2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2FPB2.txt
    Not in audiodict: M46C1FS.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FS.txt
    Not in audiodict: M23Q2VW.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VW.txt
    Not in audiodict: M56M2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT1.txt
    Not in audiodict: M56M2VT37.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT37.txt
    Not in audiodict: F56F1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT9.txt
    Not in audiodict: M46C1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI3.txt
    Not in audiodict: M56M1VI36.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI36.txt
    Not in audiodict: F27B1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FPA1.txt
    Not in audiodict: F56F1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI3.txt
    Not in audiodict: F56F1VI13.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI13.txt
    Not in audiodict: F56F2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT9.txt
    Not in audiodict: F26A1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT2.txt
    Not in audiodict: F26A1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT10.txt
    Not in audiodict: F29J1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT11.txt
    Not in audiodict: F29J1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI10.txt
    Not in audiodict: M56M2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2FPB2.txt
    Not in audiodict: F29J2VS.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VS.txt
    Not in audiodict: M23Q1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT10.txt
    Not in audiodict: M56M2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT3.txt
    Not in audiodict: F29J1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI8.txt
    Not in audiodict: M23Q2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT12.txt
    Not in audiodict: F26A1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT7.txt
    Not in audiodict: M56M2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT7.txt
    Not in audiodict: F27B2VY.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VY.txt
    Not in audiodict: M23Q1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI7.txt
    Not in audiodict: F29J2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2FPB2.txt
    Not in audiodict: F29J2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT1.txt
    Not in audiodict: M56M1VI30.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI30.txt
    Not in audiodict: M23Q1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI10.txt
    Not in audiodict: M23Q1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT6.txt
    Not in audiodict: M23Q2VT20.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT20.txt
    Not in audiodict: M56M1VI26.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI26.txt
    Not in audiodict: M56M1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT10.txt
    Not in audiodict: F56F2VW.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VW.txt
    Not in audiodict: F56F2VY.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VY.txt
    Not in audiodict: F29J1FS.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FS.txt
    Not in audiodict: F27B1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT10.txt
    Not in audiodict: F56F2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT5.txt
    Not in audiodict: M56M2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT6.txt
    Not in audiodict: M46C1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT4.txt
    Not in audiodict: F26A2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT3.txt
    Not in audiodict: M46C2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT2.txt
    Not in audiodict: M23Q1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT2.txt
    Not in audiodict: M23Q1VI14.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI14.txt
    Not in audiodict: F56F1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI10.txt
    Not in audiodict: F29J2VW.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VW.txt
    Not in audiodict: F56F2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT1.txt
    Not in audiodict: M56M1VI13.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI13.txt
    Not in audiodict: F27B2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT8.txt
    Not in audiodict: M46C1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT5.txt
    Not in audiodict: M23Q2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2FPB1.txt
    Not in audiodict: F29J1VI13.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI13.txt
    Not in audiodict: M23Q2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT1.txt
    Not in audiodict: F27B2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT6.txt
    Not in audiodict: F29J1VI14.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI14.txt
    Not in audiodict: M46C1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT11.txt
    Not in audiodict: M56M1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT3.txt
    Not in audiodict: F56F1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT6.txt
    Not in audiodict: F29J1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT3.txt
    Not in audiodict: M46C2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT11.txt
    Not in audiodict: M56M1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT11.txt
    Not in audiodict: F29J2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT8.txt
    Not in audiodict: F26A2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT6.txt
    Not in audiodict: M56M2VT13.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT13.txt
    Not in audiodict: M23Q1VI17.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI17.txt
    Not in audiodict: M56M1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI11.txt
    Not in audiodict: F29J1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI9.txt
    Not in audiodict: M46C1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI8.txt
    Not in audiodict: M56M2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT4.txt
    Not in audiodict: M56M2VT23.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT23.txt
    Not in audiodict: M56M1VI17.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI17.txt
    Not in audiodict: M23Q2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT4.txt
    Not in audiodict: M46C2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT6.txt
    Not in audiodict: M23Q1FY.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FY.txt
    Not in audiodict: M56M2VT39.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT39.txt
    Not in audiodict: M56M2VT31.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT31.txt
    Not in audiodict: F56F1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI11.txt
    Not in audiodict: M56M2PS.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2PS.txt
    Not in audiodict: F27B1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI8.txt
    Not in audiodict: F27B1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI11.txt
    Not in audiodict: M46C2VW.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VW.txt
    Not in audiodict: M23Q2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT10.txt
    Not in audiodict: F56F1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI9.txt
    Not in audiodict: M46C2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT10.txt
    Not in audiodict: M23Q1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI12.txt
    Not in audiodict: F29J1VI15.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI15.txt
    Not in audiodict: F29J1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI4.txt
    Not in audiodict: M56M1VI23.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI23.txt
    Not in audiodict: F26A2VS.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VS.txt
    Not in audiodict: F27B1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT8.txt
    Not in audiodict: M56M1VI27.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI27.txt
    Not in audiodict: M46C1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT1.txt
    Not in audiodict: M46C1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT8.txt
    Not in audiodict: M56M1VI20.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI20.txt
    Not in audiodict: M46C1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI11.txt
    Not in audiodict: F56F2VT13.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT13.txt
    Not in audiodict: M56M1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI10.txt
    Not in audiodict: M23Q1FS.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FS.txt
    Not in audiodict: M56M1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI8.txt
    Not in audiodict: M56M2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT5.txt
    Not in audiodict: M56M1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI1.txt
    Not in audiodict: F27B1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI4.txt
    Not in audiodict: F26A2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT5.txt
    Not in audiodict: M56M1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI4.txt
    Not in audiodict: F29J1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI1.txt
    Not in audiodict: F27B1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI12.txt
    Not in audiodict: M56M2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT8.txt
    Not in audiodict: M56M1VI25.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI25.txt
    Not in audiodict: F26A1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT5.txt
    Not in audiodict: F56F1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT11.txt
    Not in audiodict: F29J1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT4.txt
    Not in audiodict: F56F1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT2.txt
    Not in audiodict: M23Q1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI3.txt
    Not in audiodict: M56M1VI33.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI33.txt
    Not in audiodict: M56M1VI28.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI28.txt
    Not in audiodict: F29J2VT18.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT18.txt
    Not in audiodict: F27B1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI9.txt
    Not in audiodict: M23Q1VI21.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI21.txt
    Not in audiodict: M56M2VT24.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT24.txt
    Not in audiodict: F56F1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI5.txt
    Not in audiodict: M23Q1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT3.txt
    Not in audiodict: M23Q1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT9.txt
    Not in audiodict: M23Q1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FPA2.txt
    Not in audiodict: F27B2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT2.txt
    Not in audiodict: M46C1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT3.txt
    Not in audiodict: M23Q2VY.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VY.txt
    Not in audiodict: M23Q2VT21.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT21.txt
    Not in audiodict: F27B1FW.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FW.txt
    Not in audiodict: M56M1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI5.txt
    Not in audiodict: M46C2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT1.txt
    Not in audiodict: M23Q2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT3.txt
    Not in audiodict: M46C2VY.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VY.txt
    Not in audiodict: F29J1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI6.txt
    Not in audiodict: M46C2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2FPB1.txt
    Not in audiodict: M56M2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2FPB1.txt
    Not in audiodict: F56F2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT6.txt
    Not in audiodict: F27B2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT12.txt
    Not in audiodict: F29J1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT7.txt
    Not in audiodict: F27B2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT9.txt
    Not in audiodict: F29J1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI5.txt
    Not in audiodict: M23Q1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI2.txt
    Not in audiodict: F27B1FY.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FY.txt
    Not in audiodict: M56M2VT16.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT16.txt
    Not in audiodict: M46C1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FPA2.txt
    Not in audiodict: M56M2VT35.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT35.txt
    Not in audiodict: M56M1VI29.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI29.txt
    Not in audiodict: F56F2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT2.txt
    Not in audiodict: F27B2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT5.txt
    Not in audiodict: M46C1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT10.txt
    Not in audiodict: M46C1FW.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FW.txt
    Not in audiodict: F27B2PS.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2PS.txt
    Not in audiodict: M23Q2PS.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2PS.txt
    Not in audiodict: F29J2VT14.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT14.txt
    Not in audiodict: M46C2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT8.txt
    Not in audiodict: F27B1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT4.txt
    Not in audiodict: M56M1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT5.txt
    Not in audiodict: F29J1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI12.txt
    Not in audiodict: M23Q1VI13.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI13.txt
    Not in audiodict: M23Q1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI9.txt
    Not in audiodict: M46C2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT3.txt
    Not in audiodict: F27B1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT3.txt
    Not in audiodict: F27B1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT1.txt
    Not in audiodict: M56M2VT25.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT25.txt
    Not in audiodict: F56F2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT12.txt
    Not in audiodict: M56M2VT18.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT18.txt
    Not in audiodict: M56M1VI14.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI14.txt
    Not in audiodict: F29J1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT2.txt
    Not in audiodict: F29J2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT11.txt
    Not in audiodict: M56M2VT19.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT19.txt
    Not in audiodict: F56F2PS.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2PS.txt
    Not in audiodict: F26A1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FPA1.txt
    Not in audiodict: M46C1FY.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FY.txt
    Not in audiodict: M46C1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT2.txt
    Not in audiodict: F27B1FS.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FS.txt
    Not in audiodict: F27B1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI6.txt
    Not in audiodict: M46C2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT9.txt
    Not in audiodict: M56M1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT2.txt
    Not in audiodict: F26A1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI6.txt
    Not in audiodict: F26A1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI1.txt
    Not in audiodict: M56M1VI19.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI19.txt
    Not in audiodict: F29J2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT12.txt
    Not in audiodict: M23Q1FW.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FW.txt
    Not in audiodict: M56M1VI41.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI41.txt
    Not in audiodict: M56M2VT36.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT36.txt
    Not in audiodict: F26A2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT1.txt
    Not in audiodict: M56M2VY.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VY.txt
    Not in audiodict: F26A2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2FPB1.txt
    Not in audiodict: M56M2VT32.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT32.txt
    Not in audiodict: M56M2VT22.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT22.txt
    Not in audiodict: M56M2VS.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VS.txt
    Not in audiodict: F27B2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2FPB2.txt
    Not in audiodict: M56M2VT40.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT40.txt
    Not in audiodict: F29J2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT2.txt
    Not in audiodict: M56M1VI37.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI37.txt
    Not in audiodict: F26A1FY.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FY.txt
    Not in audiodict: F29J2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT7.txt
    Not in audiodict: F56F1FS.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FS.txt
    Not in audiodict: F56F1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI6.txt
    Not in audiodict: F56F1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI4.txt
    Not in audiodict: M46C1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT9.txt
    Not in audiodict: M56M2VT15.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT15.txt
    Not in audiodict: M23Q1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT5.txt
    Not in audiodict: M56M2VT29.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT29.txt
    Not in audiodict: M56M1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT7.txt
    Not in audiodict: F29J2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT6.txt
    Not in audiodict: F56F2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT7.txt
    Not in audiodict: M56M1VI18.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI18.txt
    Not in audiodict: M23Q1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT4.txt
    Not in audiodict: F26A2PS.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2PS.txt
    Not in audiodict: M46C2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT5.txt
    Not in audiodict: F29J2VT15.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT15.txt
    Not in audiodict: F56F2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT3.txt
    Not in audiodict: F29J1FY.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FY.txt
    Not in audiodict: M56M1VI15.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI15.txt
    Not in audiodict: M23Q1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI11.txt
    Not in audiodict: F27B1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI1.txt
    Not in audiodict: M56M1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT9.txt
    Not in audiodict: M23Q2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT5.txt
    Not in audiodict: M23Q1VI18.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI18.txt
    Not in audiodict: M56M2VT17.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT17.txt
    Not in audiodict: M23Q2VT14.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT14.txt
    Not in audiodict: F56F2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2FPB1.txt
    Not in audiodict: M56M1VI16.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI16.txt
    Not in audiodict: M56M1VI32.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI32.txt
    Not in audiodict: M56M1FS.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FS.txt
    Not in audiodict: M23Q2VS.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VS.txt
    Not in audiodict: M56M2VT34.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT34.txt
    Not in audiodict: M56M2VT20.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT20.txt
    Not in audiodict: M56M1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT8.txt
    Not in audiodict: M56M2VT26.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT26.txt
    Not in audiodict: F26A2VW.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VW.txt
    Not in audiodict: F56F2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT10.txt
    Not in audiodict: F26A1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI5.txt
    Not in audiodict: F26A2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2FPB2.txt
    Not in audiodict: F27B1FT2.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT2.txt
    Not in audiodict: F56F2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT8.txt
    Not in audiodict: M56M1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI2.txt
    Not in audiodict: M56M2VT30.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT30.txt
    Not in audiodict: F26A1FS.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FS.txt
    Not in audiodict: M56M1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FPA1.txt
    Not in audiodict: F56F1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI12.txt
    Not in audiodict: F27B2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT3.txt
    Not in audiodict: M23Q1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI5.txt
    Not in audiodict: M46C1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI12.txt
    Not in audiodict: M56M2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT12.txt
    Not in audiodict: M23Q2VT17.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT17.txt
    Not in audiodict: F56F1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT8.txt
    Not in audiodict: F27B2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT11.txt
    Not in audiodict: F26A1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT8.txt
    Not in audiodict: M23Q1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT1.txt
    Not in audiodict: F29J2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT10.txt
    Not in audiodict: M46C1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI9.txt
    Not in audiodict: F29J1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT6.txt
    Not in audiodict: M23Q2FPB2.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2FPB2.txt
    Not in audiodict: M56M2VT41.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT41.txt
    Not in audiodict: F29J2PS.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2PS.txt
    Not in audiodict: F29J2VT13.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT13.txt
    Not in audiodict: M23Q2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT9.txt
    Not in audiodict: M23Q2VT18.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT18.txt
    Not in audiodict: F56F1FW.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FW.txt
    Not in audiodict: M56M1FW.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FW.txt
    Not in audiodict: M23Q1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI6.txt
    Not in audiodict: M56M1FY.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FY.txt
    Not in audiodict: F27B1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI10.txt
    Not in audiodict: M46C1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI4.txt
    Not in audiodict: F26A1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT1.txt
    Not in audiodict: M23Q2VT16.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT16.txt
    Not in audiodict: F26A1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI4.txt
    Not in audiodict: M46C1VI10.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI10.txt
    Not in audiodict: M56M2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT10.txt
    Not in audiodict: F26A2VY.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VY.txt
    Not in audiodict: F29J1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FPA1.txt
    Not in audiodict: F29J1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT1.txt
    Not in audiodict: F56F1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT4.txt
    Not in audiodict: M56M1VI31.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI31.txt
    Not in audiodict: M23Q2VT13.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT13.txt
    Not in audiodict: M56M1VI39.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI39.txt
    Not in audiodict: F26A1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI2.txt
    Not in audiodict: F26A1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT6.txt
    Not in audiodict: F29J2VT5.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT5.txt
    Not in audiodict: F27B2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2FPB1.txt
    Not in audiodict: M23Q2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT2.txt
    Not in audiodict: M56M2VT38.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT38.txt
    Not in audiodict: M46C2PS.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2PS.txt
    Not in audiodict: M23Q1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT11.txt
    Not in audiodict: M23Q2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT7.txt
    Not in audiodict: F29J2VT17.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT17.txt
    Not in audiodict: M56M1VI24.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI24.txt
    Not in audiodict: F26A1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FPA2.txt
    Not in audiodict: M23Q1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI8.txt
    Not in audiodict: M56M1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT1.txt
    Not in audiodict: M56M1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT6.txt
    Not in audiodict: F56F1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT7.txt
    Not in audiodict: F29J1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT9.txt
    Not in audiodict: M23Q1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FPA1.txt
    Not in audiodict: F27B1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT6.txt
    Not in audiodict: F27B1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FPA2.txt
    Not in audiodict: M46C2VT12.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT12.txt
    Not in audiodict: M23Q2VT8.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT8.txt
    Not in audiodict: M46C1VI5.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI5.txt
    Not in audiodict: M46C1FT6.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT6.txt
    Not in audiodict: M23Q2VT6.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT6.txt
    Not in audiodict: M56M2VT14.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT14.txt
    Not in audiodict: M56M1VI22.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI22.txt
    Not in audiodict: M56M1VI34.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI34.txt
    Not in audiodict: M56M2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT11.txt
    Not in audiodict: M46C1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI7.txt
    Not in audiodict: F56F2VT11.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT11.txt
    Not in audiodict: M46C2VS.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VS.txt
    Not in audiodict: F29J1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT5.txt
    Not in audiodict: F56F2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VT4.txt
    Not in audiodict: F56F1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT5.txt
    Not in audiodict: F27B2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT7.txt
    Not in audiodict: F26A1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT3.txt
    Not in audiodict: M46C1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FT7.txt
    Not in audiodict: M46C2VT7.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT7.txt
    Not in audiodict: M23Q1VI16.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI16.txt
    Not in audiodict: F29J1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI3.txt
    Not in audiodict: F27B1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI3.txt
    Not in audiodict: F56F1FT3.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT3.txt
    Not in audiodict: M56M1VI35.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI35.txt
    Not in audiodict: F29J1VI11.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI11.txt
    Not in audiodict: M56M1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI6.txt
    Not in audiodict: F56F1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI7.txt
    Not in audiodict: F29J2FPB1.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2FPB1.txt
    Not in audiodict: F56F1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI2.txt
    Not in audiodict: F26A2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT2.txt
    Not in audiodict: F56F1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FPA2.txt
    Not in audiodict: F27B1FT5.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT5.txt
    Not in audiodict: M46C1VI6.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI6.txt
    Not in audiodict: F56F1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FPA1.txt
    Not in audiodict: F26A1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1VI3.txt
    Not in audiodict: F26A2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/F26A2VT4.txt
    Not in audiodict: M56M1VI38.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI38.txt
    Not in audiodict: M23Q1VI4.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI4.txt
    Not in audiodict: M23Q2VT19.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT19.txt
    Not in audiodict: F27B2VW.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VW.txt
    Not in audiodict: F29J2VY.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VY.txt
    Not in audiodict: F27B2VT1.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT1.txt
    Not in audiodict: M56M1VI9.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI9.txt
    Not in audiodict: F26A1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT11.txt
    Not in audiodict: F27B1FT11.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT11.txt
    Not in audiodict: M46C1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1VI1.txt
    Not in audiodict: M56M1VI21.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI21.txt
    Not in audiodict: M56M2VT33.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT33.txt
    Not in audiodict: M56M2VT27.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT27.txt
    Not in audiodict: F27B2VT10.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT10.txt
    Not in audiodict: F56F1FT1.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT1.txt
    Not in audiodict: M23Q2VT15.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q2VT15.txt
    Not in audiodict: M56M1VI3.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI3.txt
    Not in audiodict: F29J1FW.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FW.txt
    Not in audiodict: F26A1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/F26A1FT4.txt
    Not in audiodict: F29J2VT16.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT16.txt
    Not in audiodict: F29J1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI7.txt
    Not in audiodict: F29J2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT4.txt
    Not in audiodict: F56F1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI1.txt
    Not in audiodict: F29J1VI17.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI17.txt
    Not in audiodict: M56M1FT4.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FT4.txt
    Not in audiodict: F29J1VI18.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1VI18.txt
    Not in audiodict: M56M2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT9.txt
    Not in audiodict: M56M2VT2.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT2.txt
    Not in audiodict: M23Q1FT8.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT8.txt
    Not in audiodict: F56F1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1FT10.txt
    Not in audiodict: F27B1VI2.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI2.txt
    Not in audiodict: F29J2VT3.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT3.txt
    Not in audiodict: M56M1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI7.txt
    Not in audiodict: M23Q1FT7.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1FT7.txt
    Not in audiodict: F29J1FT10.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FT10.txt
    Not in audiodict: M56M1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1FPA2.txt
    Not in audiodict: M23Q1VI19.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI19.txt
    Not in audiodict: M56M2VT21.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT21.txt
    Not in audiodict: M46C1FPA1.txt
    /datb/aphasia/dutchaudio/originalUva/M46C1FPA1.txt
    Not in audiodict: F27B2VS.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VS.txt
    Not in audiodict: M46C2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/M46C2VT4.txt
    Not in audiodict: F29J1FPA2.txt
    /datb/aphasia/dutchaudio/originalUva/F29J1FPA2.txt
    Not in audiodict: M23Q1VI15.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI15.txt
    Not in audiodict: F27B1VI7.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1VI7.txt
    Not in audiodict: M56M1VI12.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI12.txt
    Not in audiodict: M23Q1VI1.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI1.txt
    Not in audiodict: F29J2VT9.txt
    /datb/aphasia/dutchaudio/originalUva/F29J2VT9.txt
    Not in audiodict: F56F1VI8.txt
    /datb/aphasia/dutchaudio/originalUva/F56F1VI8.txt
    Not in audiodict: F27B2VT4.txt
    /datb/aphasia/dutchaudio/originalUva/F27B2VT4.txt
    Not in audiodict: M56M2VT28.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VT28.txt
    Not in audiodict: F56F2VS.txt
    /datb/aphasia/dutchaudio/originalUva/F56F2VS.txt
    Not in audiodict: M56M1VI40.txt
    /datb/aphasia/dutchaudio/originalUva/M56M1VI40.txt
    Not in audiodict: M56M2VW.txt
    /datb/aphasia/dutchaudio/originalUva/M56M2VW.txt
    Not in audiodict: F27B1FT9.txt
    /datb/aphasia/dutchaudio/originalUva/F27B1FT9.txt
    Not in audiodict: M23Q1VI20.txt
    /datb/aphasia/dutchaudio/originalUva/M23Q1VI20.txt
    In: 554
    Not in: 400


<h3>Een check of downloaden echt een succes is geweest.</h3>


```python
import glob

# Folder op de server dus waar de data is gedownload
uva_data_folder = '/datb/aphasia/languagedata/uva/original/*'

# Ophalen van alle items in de folder
uva_data = glob.glob(uva_data_folder)

textfiles = 0
audiofiles = 0

# Optellen van tekst en audio items in de folder
for x in uva_data:
    dat = x.split('.')[-1]
    if 'txt' in dat:
        textfiles += 1
    else:
        audiofiles += 1

# Resultaat
print('Resultaat van de optelling:')
print('audio items: {}'.format(audiofiles))
print('text items: {}'.format(textfiles))
```

    Resultaat van de optelling:
    audio items: 681
    text items: 954

