# Forced Alignment (`TimeFrame` + `Alignment`) evaluation code

Evaluation of the alignment of the text to time positions for transcription/closed-captioning, commonly called "forced alignment".
The `evaluate.py` processes human-annotated(`.tsv`) and machine-generated(`.mmif`) and outputs the evaluation results.

## Usage
To run the evaluation code, run the following command while in the `fa_eval` directory:  
```bash
python3 evaluate.py -m <machine_dir> -g <gold_dir> -r <result_file>
```
`-g` argument is optional. If not provided, the code will use the gold text from a predefined place in the aapb-annotations repository. 
See inside the python code for the default gold text location URL.

```bash
$ python3 evaluate.py -m /fa_eval/preds@gentle-forced-aligner-wrapper@aapb-collaboration-21-nongoldtext
Individual file results:
                    GUID   DiaCov   DiaPur  D-C-P-F1   SegCov  SegPur  S-C-P-F1
cpb-aacip-507-6w96689725 0.976800 0.972071  0.974430 0.939031     1.0  0.968557
cpb-aacip-507-pc2t43js98 0.974697 0.968273  0.971474 0.936596     1.0  0.967260
cpb-aacip-507-6h4cn6zk04 0.975139 0.969976  0.972551 0.922537     1.0  0.959708
cpb-aacip-507-cf9j38m509 0.980175 0.976447  0.978307 0.948909     1.0  0.973785
cpb-aacip-507-n29p26qt59 0.973329 0.970259  0.971792 0.935249     1.0  0.966541
cpb-aacip-507-nk3610wp6s 0.973057 0.965097  0.969061 0.930397     1.0  0.963944
cpb-aacip-507-pr7mp4wf25 0.969234 0.955383  0.962259 0.929009     1.0  0.963198
cpb-aacip-507-1v5bc3tf81 0.977150 0.970972  0.974051 0.937488     1.0  0.967735
cpb-aacip-507-r785h7cp0z 0.979036 0.971423  0.975215 0.946178     1.0  0.972345
cpb-aacip-507-zk55d8pd1h 0.971831 0.960442  0.966103 0.930962     1.0  0.964247
cpb-aacip-507-vm42r3pt6h 0.977702 0.969733  0.973701 0.939191     1.0  0.968642
cpb-aacip-507-7659c6sk7z 0.975849 0.966133  0.970967 0.922775     1.0  0.959837
cpb-aacip-507-v40js9j432 0.975791 0.970891  0.973335 0.937696     1.0  0.967846
cpb-aacip-507-4746q1t25k 0.974794 0.964196  0.969466 0.932502     1.0  0.965072
cpb-aacip-507-9882j68s35 0.964569 0.958660  0.961606 0.925507     1.0  0.961313
cpb-aacip-507-zw18k75z4h 0.967243 0.958316  0.962759 0.933812     1.0  0.965773
cpb-aacip-507-154dn40c26 0.974231 0.964671  0.969428 0.935661     1.0  0.966761
cpb-aacip-507-v11vd6pz5w 0.980989 0.977637  0.979310 0.942218     1.0  0.970250
cpb-aacip-507-4t6f18t178 0.969985 0.959975  0.964954 0.928064     1.0  0.962690


Average results:
DiaCov      0.974295
DiaPur      0.966871
D-C-P-F1    0.970567
SegCov      0.934410
SegPur      1.000000
S-C-P-F1    0.966079
dtype: float64
```

## Important Notes / Common Issues
* There is some [issue](https://github.com/clamsproject/aapb-evaluations/issues/31) with the gold vs silver/[non-gold text](aapb-collaboration-21-nongoldtext) in the fa_eval. 
