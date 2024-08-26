# ASR Evaluation Report  -- Distil-Whisper `small`

### App
1. [distil-whisper-wrapper v1.1](https://github.com/clamsproject/app-distil-whisper-wrapper/tree/v1.1) is used to generated the preds MMIF files from 20 videos. For this evaluation, MMIF files are generated through the distil whisper `small` model.
2. Preds MMIF files can be found [here](https://github.com/clamsproject/aapb-evaluations/tree/asr-new-eval/asr_eval/preds%40distil-whisper-wrapper-small%40aapb-collaboration-21).
* **Note**: `cpb-aacip-507-vd6nz81n6r.video` does not exist, therefore no MMIF file is generated. The total number of valid MMIF files is **19**, **NOT** 20.

### Evaluation Code
The evaluation code can be found [here](https://github.com/clamsproject/aapb-evaluations/tree/asr-new-eval/asr_eval)

### Evaluation Metric
WER (Word Error Rate) is used as the evaluation metric that is implemented by 
the above mentioned evaluation code. WER calculates the accuracy of Automatic 
Speech Recognition (ASR) on the word level. To get a WER, the number of errors 
is divided by the number of total words spoken. In other words, WER tells 
"how wrong" the predicted result can be. Therefore, a smaller WER indicates a 
better performance. More information can be found [here](https://en.wikipedia.org/wiki/Word_error_rate).
The `jiwer` [package](https://jitsi.github.io/jiwer/) has a supports WER 
calculations and is used in our evaluation code.

### Evaluation Dataset
Gold standard annotations are located [here](https://github.com/clamsproject/aapb-collaboration/tree/89b8b123abbd4a9a67c525cc480173b52e0d05f0/21), with file name starting with the corresponding video IDs.

### Evaluation Results
We are comparing it to our results from the [report-20240802-preds@whisper-wrapper-small@aapb-collaboration-21](https://github.com/clamsproject/aapb-evaluations/blob/asr-new-eval/asr_eval/report-20240802-preds%40whisper-wrapper-small%40aapb-collaboration-21.md)
in order to determine which ASR model of the same size yields better results. 
Additionally, we evaluate MMIF files with and without case sensitivity.
>1. **Case-sensitive distil-whisper (CaseSD)**: Upper case and lower case are treated differently using the dist-whisper model, e.g. Apple ≠ apple.
>2. **Case Insensitive distil-whisper (CaseID)**: The transcripts from both gold and preds are capitalized using the distil-whisper model, thus making case insignificant, e.g. APPLE = APPLE.
>3. **Case-sensitive whisper (CaseSW)**: Using the whisper model, e.g. Apple ≠ apple.
>4. **Case Insensitive whisper (CaseIW)**: Using the whisper model, e.g. APPLE = APPLE.

These 4 conditions generate 4 different WERs.

#### A brief summary
1. The lowest WER is **0.12475920679886686**, from `cpb-aacip-507-r785h7cp0z`, CaseIW.
2. The highest WER is **0.3545075031027869**, from `cpb-aacip-507-n29p26qt59`, CaseSW.
3. The majority of the whisper-wrapper WERs fall within 15% ~ 30% range, and the majority of the distil-whisper-wrapper WERs fall within the 20%-30% range.
4. When ignoring the case, **ALL** of the WERs become slighly lower, indicating a slightly higher accuracy. This is true for both models.
5. In general, whisper would have a slightly higher accuracy than distil-whisper, but distil-whisper appeared to be more consistent with its results.
5. The avarage WER among 19 MMIF files are the following. Whisper performs a little better, but they are statistically comparable:

    | CaseSW |       CaseIW        |        CaseSD        |      Case SD     |
    |:-------------------:|:--------------------:|:----------------:| :----------------: |
    | 0.21172602739726026| 0.19616438356164384 |  0.2249241428933816  | 0.20694108582670986 |

#### Full Results
| small                    | CaseSW              | CaseIW              | Case SD             | Case ID             |
|--------------------------|---------------------|---------------------|---------------------|---------------------|
| cpb-aacip-507-vm42r3pt6h | 0.1681503461918892  | 0.1554154302670623  | 0.2000737372496006  | 0.18422022858547377 |
| cpb-aacip-507-zk55d8pd1h | 0.22514511547486724 | 0.2043966901321477  | 0.3093746565556655  | 0.2946477634904935  |
| cpb-aacip-507-zw18k75z4h | 0.18701668701668703 | 0.17399267399267399 | 0.21456016177957532 | 0.19939332659251768 |
| cpb-aacip-507-154dn40c26 | 0.24619995718261614 | 0.2010276172125883  | 0.1761624099541585  | 0.15913555992141454 |
| cpb-aacip-507-1v5bc3tf81 | 0.1867215302491103  | 0.16814946619217083 | 0.21234540636042404 | 0.1963339222614841  |
| cpb-aacip-507-4746q1t25k | 0.2214220393232739  | 0.1979881115683585  | 0.2292753301520364  | 0.20408389745866162 |
| cpb-aacip-507-4t6f18t178 | 0.19594320486815417 | 0.1817444219066937  | 0.2510451921162652  | 0.23312761298029067 |
| cpb-aacip-507-6h4cn6zk04 | 0.18592964824120603 | 0.1661641541038526  | 0.20432692307692307 | 0.1830201048951049  |
| cpb-aacip-507-6w96689725 | 0.2092817679558011  | 0.19116022099447513 | 0.19547454431175362 | 0.17347580138277813 |
| cpb-aacip-507-7659c6sk7z | 0.21949052132701422 | 0.20882701421800948 | 0.23309395571514063 | 0.2211250748055057  |
| cpb-aacip-507-9882j68s35 | 0.297281993204983   | 0.25254813137032844 | 0.23622497616777885 | 0.2204003813155386  |
| cpb-aacip-507-cf9j38m509 | 0.22393462970624742 | 0.2056268100951593  | 0.28209000302023557 | 0.2642706131078224  |
| cpb-aacip-507-n29p26qt59 | 0.3545075031027869  | 0.3371318966489902  | 0.24928803905614322 | 0.2265052888527258  |
| cpb-aacip-507-nk3610wp6s | 0.1610952276226083  | 0.145150648779415   | 0.19429833789185777 | 0.1797811908268462  |
| cpb-aacip-507-pc2t43js98 | 0.20793565683646112 | 0.1899195710455764  | 0.23735126745990687 | 0.2182100362131402  |
| cpb-aacip-507-pr7mp4wf25 | 0.19402277039848198 | 0.17303130929791272 | 0.23008233793343383 | 0.20746839846921025 |
| cpb-aacip-507-r785h7cp0z | 0.13552407932011332 | 0.12475920679886686 | 0.17382966723068247 | 0.16097010716300056 |
| cpb-aacip-507-v11vd6pz5w | 0.2353726362625139  | 0.21590656284760845 | 0.26609766637856525 | 0.24632670700086431 |
| cpb-aacip-507-v40js9j432 | 0.21172602739726026 | 0.19616438356164384 | 0.17856410256410257 | 0.15938461538461537 |

### Limitations/Issues
1. A considerable number of useless strings are found in all of the gold files. These files include titles such as `INFO`, `News Summary` as well as each speaker's name along with every utterance. These are useful information for readers, but are, in reality, non-existent in the actual speech. This significantly affects the evaluation **negatively**. Some processing needs to be done on gold files.
2. On the other hand, some of the gold files are missing preview, summary, and add content in the video, whereas whisper preserves this information. This disprepancy also has a significant **negative** impact on evaluation results. Manually removing the extra audio not found in the golds from our preds as much as halves the WER score.
4. This [iteration](https://github.com/clamsproject/app-distil-whisper-wrapper/tree/HF-pipeline-fixed-timestamps) of the distil-whisper app often provides a warning suggesting that transcription may have been cut off in the middle of a word, and even though no cases have been found where the audio was cut off, this currently would not efect the accuracy becasue of the extra audio in the file that's not in the gold data anyways.
5. Considering `small` model's size, WER is expected to decrease in the future when transcribing with bigger models.
