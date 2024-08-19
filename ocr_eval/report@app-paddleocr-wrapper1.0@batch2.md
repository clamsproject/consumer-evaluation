# Paddle OCR Evaluation Report

## App

The `preds` MMIF files were generated using the [PaddleOCR Wrapper](https://github.com/clamsproject/app-paddleocr-wrapper). 

## Evaluation Dataset

The gold annotation data can be found [here](https://github.com/clamsproject/aapb-annotations/tree/f96f857ef83acf85f64d9a10ac8fe919e06ce51e/newshour-chyron/golds/batch2).

## Evaluation Code

The script used for evaluation can be found [here](https://github.com/clamsproject/aapb-evaluations/blob/asr_eval_update/ocr_eval/evaluate.py).

## Metrics

This dataset was evaluated using `Character Error Rate` , or `CER`. CER is a measurement of the accuracy of the predictions - it is the proportion of *incorrectly* predicted characters, calculated using edit distance. Edit distance can be thought of as *"the least number of operations required to convert one string to another, where an operation can be one of substitution, deletion or insertion."*

As such, a higher CER represents a more inaccurate prediction, with perfect prediction being 0.0. Ideal CER performance is below 10% (<0.1).

### Alignment

Unlike parseqocr OCR and tessract OCR but similar to docTR OCR, PaddleOCR has `TextDocuments` annotation which is anchored to `TimePoint` annotations. The evaluation script aligns the `TextDocument` to the `TimeFrame` annotations in the gold standard data.

### Results~~~~

``` 
cpb-aacip-507-154dn40c26:	0.12860753715914838
cpb-aacip-507-1v5bc3tf81:	0.13939197862456584
cpb-aacip-507-4t6f18t178:	0.13598766821351918
cpb-aacip-507-6w96689725:	0.052824508088330425
cpb-aacip-507-7659c6sk7z:	0.011904762436946234
cpb-aacip-507-9882j68s35:	0.0269739522288243
cpb-aacip-507-bz6154fc44:	0.07854265886647947
cpb-aacip-507-cf9j38m509:	0.41159106853107613
cpb-aacip-507-m61bk17f5g:	0.13168174948762446
cpb-aacip-507-n29p26qt59:	0.06718578487634659
cpb-aacip-507-nk3610wp6s:	0.17419327065348625
cpb-aacip-507-pc2t43js98:	0.06393448815991482
cpb-aacip-507-pr7mp4wf25:	0.22759244378123963
cpb-aacip-507-r785h7cp0z:	0.1741071457141324
cpb-aacip-507-v11vd6pz5w:	0.21652369061484933
cpb-aacip-507-v40js9j432:	0.25155910663306713
cpb-aacip-507-vm42r3pt6h:	0.067090870346874
cpb-aacip-507-zk55d8pd1h:	0.20503402673281157
cpb-aacip-507-zw18k75z4h:	0.2358576183517774
cpb-aacip-525-028pc2v94s:	0.7665860783308744
cpb-aacip-525-3b5w66b279:	0.16173261361053357
cpb-aacip-525-9g5gb1zh9b:	0.46138236878646743
cpb-aacip-525-bg2h70914g:	0.2513653819914907
Total Mean CER:	0.1931152509661035

```
### Side-by-Side Views

The side-by-side comparison of gold and test annotations is visible within the [`results` output](https://github.com/clamsproject/aapb-evaluations/tree/asr_eval_update/ocr_eval/results%40app-paddleocr-wrapper1.0%40batch2).
Each file consists of the annotations for one video document,a which look like the following:

```json
"516.5271": {
    "ref_text": "rita lavelle former e.p.a. official",
    "hyp_text": "rita lavelle former e.p.a.official",
    "cer": 0.02857142873108387
}
```

### Limitations

PaddleOCR creates a structured output with only line object, where there is only one level of bounding box annotation. The smallest recognized unit is `SENRENCE` annotation, which usually contains a line of words of a block of words. The `TextDocuments` annotation is created by appended all the `SENTENCE` unites at a timepoint together with `\n` in between. All newlines are stripped from the gold and hypothesis data so the organization.

The CER is also inflated by PaddleOCR becuase PaddleOCR detects and extracts text from other parts of the scene, such as watermarks, logos, and other non-relevant and not annotated on-screen text. 
For example, in the following case:
```
"785.7771": {
    "ref_text": "mayor ed koch",
    "hyp_text": "aug.30,1985 mayor ed-koch",
    "cer": 1.0
  },
```
`aug.30,1985` is obviously a watermarks of date that is considered irrelevance not annotated in gold data. Since PaddleOCR don't have the function of only extracting key parts, it dectects all the text in an image indiscriminately, causing the hyp_text has more content and text than the gold data. This issue significantly increase the `cer` number, even to an extreme case like 1.0 cer. 
Since in the both evaluatation of doctrOCR and Paddleocr this issue results a failure of presenting the real quality of ocr output by the ocr tool, a new assessment calculation or renewal version of gold data may be necessary.
