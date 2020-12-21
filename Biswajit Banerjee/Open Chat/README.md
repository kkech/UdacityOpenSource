# Open Chat
This repository contains the showcase project developed in the Secure and Private AI Scholarship. For more details, please visit this site

Author: Biswajit Banerjee <br>
Email : sumonbanner@gmail.com <br>
Slack : @Biswajit Banerjee  

## Introduction

This is a messaging application built totally from scratch, where users from multiple locations in the globe can join and chat with each other. The special factor in this application is, it translates the message from sending user to the receiving user's choice.<br>
Let's say, a user Bob from London is chatting with another user Julia from Germany. Bob prefers to chat in English and Julia prefers to chat in German. So when both of the logs into this application and chats with each other <br>Bob's `English messages will be translated to German` and will be sent to Julia and Julia's `German messages will be translated to English` and will be sent to Bob.<br>
Currently it supports English (en) and German (de) only, but there is a lot of room for adding other languages.


## Definitions
<b>Neural Machine Translation (NMT) </b>: *Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems*. <br>
>"In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language"
— Page 98, Deep Learning, 2016. 
<br>

<b>Socket</b>: *A socket is one endpoint of a two-way communication link between two programs running on the network*.
> In simpler terms, a socket is a network object the helps sending and reciving message over the internet

## Requirements

* Python 3.7 or above
* PyTorch 1.1.0 or above
* PySyft 0.1 or above (optional)
* Matplotlib 3.0 or above (optional)

## Running
This program allows you to send messages directly from your command line to the internet and connect to a server located anywhere.<br>
Fisrt we will need to start the server<br>
```bash
python3 server.py -host 127.0.0.1 -port 1234
```
Then once the server is up and running the users will be abele to join<br>
```bash
python3 client.py -host 127.0.0.1 -port 1234
```

## Results
First the Seq2Seq machine translation results<br>
```
Input :  Wir sind noch nicht ganz fertig.
Target:  we re not totally ready yet.
Output:  we re not all finished yet.
```
```
Input :  Ich bin sonntags nicht zu hause.
Target:  i m not home on sundays.
Output:  i m not home on sundays.
```
```
Input :  Ich lerne maschineschreiben.
Target:  i m learning how to type.
Output:  i m studying how to type.
```
```
Input :  Ich bin nicht mehr hungrig.
Target:  i m not hungry anymore.
Output:  i m not hungry anymore.
```
```
Input :  Ich uberfliege gerade seinen bericht.
Target:  i m skimming his report right now.
Output:  i m skimming his report right now.
```
```
Input :  Ich bin froh , bei dir zu sein.
Target:  i am glad to be with you.
Output:  i am glad i am with you.

```
Some visuals of the entire users and server <br>
[Example](./files/example2.png?raw=true 'Example')
More can be found inside files folder.



## Conclusions
This application is primarily based on network communications and message translations. It also has a lot of space to add as many languages as we can which in future will become a big application. As the sockets are very efficient multiple language translations will just add very little impact to this application.<br> 
It uses PyTorch for translations which I learnt from this course and sockets which I also learnt from this course.
I am very much thankful to Udacity for providing such a huge platform and Facebook for providing the scholarship.<br>
This project will not be possible without Andrew Trask's guidance in this course and Palak Sadani & Akshit Jain, both of their constant help and supervision, Thanks a lot.<br>
Last but not least the Secure and Private AI slack community always ready to help, thank all of you.