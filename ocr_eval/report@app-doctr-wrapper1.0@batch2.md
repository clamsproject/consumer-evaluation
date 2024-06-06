# docTR OCR Evaluation Report

## App

The `preds` MMIF files were generated using the [docTR OCR Wrapper](http://apps.clams.ai/app-doctr-wrapper/v1.0/). 

## Evaluation Dataset

The gold annotation data can be found [here](https://github.com/clamsproject/aapb-annotations/tree/f96f857ef83acf85f64d9a10ac8fe919e06ce51e/newshour-chyron/golds/batch2).

## Evaluation Code

The script used for evaluation can be found [here](https://github.com/clamsproject/aapb-evaluations/blob/doctr-eval/ocr_eval/evaluate.py).

## Metrics

This dataset was evaluated using `Character Error Rate` , or `CER`. CER is a measurement of the accuracy of the predictions - it is the proportion of *incorrectly* predicted characters, calculated using edit distance. Edit distance can be thought of as *"the least number of operations required to convert one string to another, where an operation can be one of substitution, deletion or insertion."*

As such, a higher CER represents a more inaccurate prediction, with perfect prediction being 0.0. Ideal CER performance is below 10% (<0.1).

### Alignment

Unlike previous OCR apps which have been evaluated, the docTR OCR `TextDocuments` are anchored to `TimePoint` annotations. The evaluation script aligns the `TextDocument` to the `TimeFrame` annotations in the gold standard data.

### Results~~~~

``` 
cpb-aacip-507-154dn40c26:	0.1456622997242393
cpb-aacip-507-1v5bc3tf81:	0.05068027281335422
cpb-aacip-507-4t6f18t178:	0.14087630063295364
cpb-aacip-507-6w96689725:	0.02311990720530351
cpb-aacip-507-7659c6sk7z:	0.0535714291036129
cpb-aacip-507-9882j68s35:	0.01698412708938122
cpb-aacip-507-bz6154fc44:	0.08259188593365252
cpb-aacip-507-cf9j38m509:	0.024315853349187157
cpb-aacip-507-m61bk17f5g:	0.07906963215454628
cpb-aacip-507-n29p26qt59:	0.06396528871523009
cpb-aacip-507-nk3610wp6s:	0.23613259137015452
cpb-aacip-507-pc2t43js98:	0.06444251034408807
cpb-aacip-507-pr7mp4wf25:	0.006946386912694344
cpb-aacip-507-r785h7cp0z:	0.15684234762662336
cpb-aacip-507-v11vd6pz5w:	0.13520408208881105
cpb-aacip-507-v40js9j432:	0.23942378703504802
cpb-aacip-507-vm42r3pt6h:	0.05793061244644617
cpb-aacip-507-zk55d8pd1h:	0.18929834264729703
cpb-aacip-507-zw18k75z4h:	0.33453476103022695
cpb-aacip-525-028pc2v94s:	0.7361944615840912
cpb-aacip-525-3b5w66b279:	0.029640428189720427
cpb-aacip-525-9g5gb1zh9b:	0.2090792879462242
cpb-aacip-525-bg2h70914g:	0.35715920602281886
Total Mean CER:	0.14928981747676978

```
### Side-by-Side Views

The side-by-side comparison of gold and test annotations is visible within the [`results` output](https://github.com/clamsproject/aapb-evaluations/tree/doctr-eval/ocr_eval/results%40app-doctr-wrapper1.0%40batch2).
Each file consists of the annotations for one video document,a which look like the following:

```json
"2188.2771": {
    "ref_text": "craig watkins district attorney, dallas county",
    "hyp_text": "craig watkins district attorney, dallas county kqed9 2",
    "cer": 0.17391304671764374
  }
```

### Limitations

docTR creates a structured output with word, line, and block objects. This facilitates a reading order
that should generally, but not necessarily, be aligned with the intuitions of annotators. In `TextDocuments` lines are delineated
by `\n` and blocks are delineated by `\n\n`. All newlines are stripped from the gold and hypothesis data so the organization
of these bounding boxes are currently not being evaluated.

The distinctions between these bounding boxes for lines and blocks aka `sentences` and `paragraphs` in the MMIF are relevant
for apps down the pipeline like the RFB app which uses lines/blocks to help delineate roles from fillers, which tend to
exist on separate lines in chyrons.

The CER is also inflated (made to look worse) by docTR detecting and extracting text from other parts of the scene,
watermarks, logos, and other non-relevant and not annotated on-screen text. See `"kqed9 2"` in example side-by-side result.