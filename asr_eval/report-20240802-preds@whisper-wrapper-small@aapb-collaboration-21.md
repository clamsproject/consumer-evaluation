# ASR Evaluation Report  -- Whisper `small`

### App
1. [whisper-wrapper v8](https://github.com/clamsproject/app-whisper-wrapper/tree/v8)  is used to generated the preds MMIF files from 20 videos. For this evaluation, MMIF files are generated through whisper `small` model.
2. Preds MMIF files can be found [here](https://github.com/clamsproject/aapb-evaluations/tree/asr-new-eval/asr_eval/preds%40whisper-wrapper-small%40aapb-collaboration-21).
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

### Evaluation results: 
We evaluate MMIF files under 2 conditions:
>1. **Case-sensitive (CaseS)**: Upper case and lower case are treated differently, e.g. Apple ≠ apple. 
>2. **Case Insensitive (CaseI)**: The transcripts from both gold and preds are capitalized, thus making case insignificant, e.g. APPLE = APPLE.

These 2 conditions generate 2 different WERs.

#### A brief summary
1. The lowest WER is **0.12475920679886686**, from `cpb-aacip-507-r785h7cp0z`, CaseI.
2. The highest WER is **0.3545075031027869**, from `cpb-aacip-507-n29p26qt59`, CaseS.
3. The majority of the WERs fall within 15% ~ 25% range.
4. When ignoring the case, **ALL** of the WERs become slighly lower, indicating a slightly higher accuracy.
5. The avarage WER among 19 MMIF files are:
   
    | CaseS | CaseI |
    |:-----:| :---: |
    | 0.2140369127201092 |   0.19416338531755442    |

#### Full Results
| small                    | CaseS              | CaseI              |
|--------------------------|---------------------|---------------------|
| cpb-aacip-507-vm42r3pt6h | 0.1681503461918892  | 0.1554154302670623  |
| cpb-aacip-507-zk55d8pd1h | 0.22514511547486724 | 0.2043966901321477  |
| cpb-aacip-507-zw18k75z4h | 0.18701668701668703 | 0.17399267399267399 |
| cpb-aacip-507-154dn40c26 | 0.24619995718261614 | 0.2010276172125883  |
| cpb-aacip-507-1v5bc3tf81 | 0.1867215302491103  | 0.16814946619217083 |
| cpb-aacip-507-4746q1t25k | 0.2214220393232739  | 0.1979881115683585  |
| cpb-aacip-507-4t6f18t178 | 0.19594320486815417 | 0.1817444219066937  |
| cpb-aacip-507-6h4cn6zk04 | 0.18592964824120603 | 0.1661641541038526  |
| cpb-aacip-507-6w96689725 | 0.2092817679558011  | 0.19116022099447513 |
| cpb-aacip-507-7659c6sk7z | 0.21949052132701422 | 0.20882701421800948 |
| cpb-aacip-507-9882j68s35 | 0.297281993204983   | 0.25254813137032844 |
| cpb-aacip-507-cf9j38m509 | 0.22393462970624742 | 0.2056268100951593  |
| cpb-aacip-507-n29p26qt59 | 0.3545075031027869  | 0.3371318966489902  |
| cpb-aacip-507-nk3610wp6s | 0.1610952276226083  | 0.145150648779415   |
| cpb-aacip-507-pc2t43js98 | 0.20793565683646112 | 0.1899195710455764  |
| cpb-aacip-507-pr7mp4wf25 | 0.19402277039848198 | 0.17303130929791272 |
| cpb-aacip-507-r785h7cp0z | 0.13552407932011332 | 0.12475920679886686 |
| cpb-aacip-507-v11vd6pz5w | 0.2353726362625139  | 0.21590656284760845 |
| cpb-aacip-507-v40js9j432 | 0.21172602739726026 | 0.19616438356164384 |

### Limitations/Issues
1. A considerable number of useless strings are found in all of the gold files. These files include titles such as `INFO`, `News Summary` as well as each speaker's name along with every utterance. These are useful information for readers, but are, in reality, non-existent in the actual speech. This significantly affects the evaluation **negatively**. Some processing needs to be done on gold files.
2. On the other hand, some of the gold files are missing preview, summary, and add content in the video, whereas whisper preserves this information. This disprepancy also has a significant **negative** impact on evaluation results. 
4. Considering `small` model's size, WER is expected to decrease in the future when transcribing with bigger models.