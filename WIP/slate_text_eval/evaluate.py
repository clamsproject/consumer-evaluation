import json
import re
from pathlib import Path
from mmif import Mmif, View
from typing import Dict, List, Tuple
from jiwer import wer, cer
from collections import defaultdict
from dateutil import parser
from datetime import datetime

# Define the fields we want to extract from transcriptions
FIELDS_OF_INTEREST = {
    "PROGRAM-TITLE",
    "EPISODE-TITLE",
    "SERIES-TITLE",
    "TITLE",
    "EPISODE-NO",
    "CREATE-DATE",
    "AIR-DATE",
    "DATE",
    "DIRECTOR",
    "PRODUCER",
    "CAMERA"
}

def load_gold_standard(file_path):
    """
    Load and parse the gold standard data from img_arr_prog.js
    Returns a dictionary mapping image filenames to their raw and structured annotations
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract just the array content
    content = content.strip()
    content = content.replace('imgArray = ', '')
    content = content.rstrip(';')
    
    # Parse each row into a list
    gold_standard = {}
    
    try:
        # Use json.loads to parse the array
        data = json.loads(content)
        
        # Process each entry
        for entry in data:
            filename = entry[0]  # First element is filename
            raw_transcription = entry[5]  # Sixth element is raw transcription
            structured_text = entry[6]  # Seventh element is structured transcription
            
            # Parse the structured text into key-value pairs for fields of interest
            structured_transcription = {}
            for line in structured_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()  # Convert key to uppercase
                    value = value.strip()
                    if key in FIELDS_OF_INTEREST:
                        structured_transcription[key] = value
            
            gold_standard[filename] = {
                "raw_transcription": raw_transcription,
                "structured_transcription": structured_transcription
            }
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
        
    return gold_standard

def parse_llava_text(text: str) -> Dict[str, str]:
    """
    Parse LLaVA output text to extract prompt and response
    Returns a dictionary with 'prompt' and 'response' keys
    """
    # Find content between [INST] and [/INST] tags for prompt
    prompt_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
    prompt = prompt_match.group(1).strip() if prompt_match else ""
    
    # Get everything after [/INST] for response
    response_match = re.search(r'\[/INST\](.*?)$', text, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""
    
    return {
        "prompt": prompt,
        "response": response
    }

def load_predictions(predictions_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load prediction MMIF files and extract TextDocument annotations
    Returns a dictionary mapping image filenames to lists of prediction dictionaries
    """
    predictions = {}
    pred_dir = Path(predictions_dir)
    
    for mmif_file in pred_dir.glob("*.mmif"):
        # Get corresponding image filename (replace .mmif with .jpg)
        image_filename = mmif_file.name.replace('.mmif', '.jpg')
        
        # Load and parse MMIF file
        with open(mmif_file, 'r') as f:
            try:
                mmif_data = Mmif(f.read())
            except Exception as e:
                print(f"Error loading MMIF file {mmif_file}: {e}")
                continue
        
        # Extract and parse text values from TextDocument annotations
        parsed_predictions = []
        for view in mmif_data.views:
            for annotation in view.annotations:
                if annotation.at_type == "http://mmif.clams.ai/vocabulary/TextDocument/v1":
                    parsed_text = parse_llava_text(annotation.properties.text_value)
                    parsed_predictions.append(parsed_text)
        
        if parsed_predictions:
            predictions[image_filename] = parsed_predictions
    
    return predictions

def evaluate_raw_transcription(gold_text: str, pred_text: str) -> Dict[str, float]:
    """
    Evaluate raw transcription using CER and WER metrics
    """
    # Clean up texts - remove extra whitespace and normalize
    gold_text = ' '.join(gold_text.split())
    pred_text = ' '.join(pred_text.split())
    
    return {
        "cer": cer(gold_text, pred_text),
        "wer": wer(gold_text, pred_text)
    }

def parse_structured_prediction(text: str) -> Dict[str, str]:
    """
    Parse the JSON prediction from the model's response
    """
    if not text:
        return {}
        
    # Remove ```json and ``` markers and any surrounding whitespace/newlines
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    text = re.sub(r'\s*`+\s*$', '', text)  # Remove any trailing backticks
    text = text.strip()
    
    try:
        result = json.loads(text)
        # Ensure all values are strings or None
        return {k: str(v) if v is not None else None for k, v in result.items()}
    except json.JSONDecodeError as e:
        print(f"Warning: Skipping invalid JSON prediction: {e}")
        return {}

def normalize_date(date_str: str) -> str:
    """
    Normalize date strings to a standard format (YYYY-MM-DD)
    Returns original string if it can't be parsed
    """
    if not date_str:
        return date_str
        
    try:
        # Parse the date string using dateutil
        parsed_date = parser.parse(date_str)
        # Return standardized format
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        # Return original if parsing fails
        return date_str

def evaluate_structured_fields(gold_dict: Dict[str, str], 
                             pred_dict: Dict[str, str]) -> Dict[str, float]:
    """
    Evaluate structured field extraction using accuracy
    """
    correct = 0
    total = 0
    
    # Check each field in the gold standard
    for field in FIELDS_OF_INTEREST:
        if field in gold_dict:
            total += 1
            gold_value = gold_dict[field]
            pred_value = pred_dict.get(field.lower(), '')
            
            # Normalize dates for date-related fields
            if field in {'CREATE-DATE', 'AIR-DATE', 'DATE'}:
                gold_value = normalize_date(gold_value)
                pred_value = normalize_date(pred_value)
            
            if field.lower() in pred_dict and pred_value and gold_value.lower() == pred_value.lower():
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def evaluate_dates(gold_dict: Dict[str, str], pred_dict: Dict[str, str]) -> Dict[str, int]:
    """
    Evaluate date matching across all fields, regardless of field names
    """
    # Extract and normalize all dates from gold standard
    gold_dates = set()
    for value in gold_dict.values():
        normalized_date = normalize_date(value)
        try:
            # Only add if it's actually a date
            datetime.strptime(normalized_date, '%Y-%m-%d')
            gold_dates.add(normalized_date)
        except (ValueError, TypeError):
            continue

    # Extract and normalize all dates from predictions
    pred_dates = set()
    incorrect_dates = set()  # Track incorrect dates
    for value in pred_dict.values():
        normalized_date = normalize_date(value)
        try:
            datetime.strptime(normalized_date, '%Y-%m-%d')
            pred_dates.add(normalized_date)
            if normalized_date not in gold_dates:
                incorrect_dates.add(normalized_date)
        except (ValueError, TypeError):
            continue

    return {
        "correct_dates": len(gold_dates & pred_dates),  # intersection
        "total_gold_dates": len(gold_dates),
        "incorrect_pred_dates": len(pred_dates - gold_dates),  # dates in pred but not in gold
        "incorrect_dates_examples": list(incorrect_dates)  # Add examples of incorrect dates
    }

def evaluate_non_date_fields(gold_dict: Dict[str, str], pred_dict: Dict[str, str]) -> Dict[str, int]:
    """
    Evaluate non-date field matching across all fields, regardless of field names
    """
    # Extract all non-date values from gold standard
    gold_values = set()
    date_fields = {'CREATE-DATE', 'AIR-DATE', 'DATE'}
    
    for field, value in gold_dict.items():
        if field not in date_fields and value and isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value:
                gold_values.add(cleaned_value.lower())
    
    # Extract all non-date values from predictions
    pred_values = set()
    incorrect_values = set()
    
    for field, value in pred_dict.items():
        if value and isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value:
                # Skip if it looks like a date
                try:
                    datetime.strptime(normalize_date(cleaned_value), '%Y-%m-%d')
                    continue
                except (ValueError, TypeError):
                    normalized_value = cleaned_value.lower()
                    pred_values.add(normalized_value)
                    if normalized_value not in gold_values:
                        incorrect_values.add(cleaned_value)  # Keep original case for display
    
    return {
        "correct_values": len(gold_values & {v.lower() for v in pred_values}),  # intersection
        "total_gold_values": len(gold_values),
        "total_pred_values": len(pred_values),
        "incorrect_values": len(incorrect_values),
        "incorrect_values_examples": list(incorrect_values)
    }

def evaluate_predictions(gold_data: Dict, pred_data: Dict) -> Dict:
    """
    Evaluate all predictions against gold standard
    """
    results = {
        "raw_transcription": defaultdict(list),
        "structured_fields": defaultdict(list),
        "dates": defaultdict(list),
        "non_date_fields": defaultdict(list),
        "overall": {},
        "incorrect_dates_all": set(),
        "incorrect_values_all": set()
    }
    
    # Evaluate each image
    for image_file in set(gold_data.keys()) & set(pred_data.keys()):
        gold = gold_data[image_file]
        preds = pred_data[image_file]
        
        if len(preds) >= 1:
            raw_metrics = evaluate_raw_transcription(
                gold["raw_transcription"],
                preds[0]["response"]
            )
            for metric, value in raw_metrics.items():
                results["raw_transcription"][metric].append(value)
        
        if len(preds) >= 2:
            pred_struct = parse_structured_prediction(preds[1]["response"])
            
            # Existing evaluations
            struct_metrics = evaluate_structured_fields(
                gold["structured_transcription"],
                pred_struct
            )
            for metric, value in struct_metrics.items():
                results["structured_fields"][metric].append(value)
            
            date_metrics = evaluate_dates(
                gold["structured_transcription"],
                pred_struct
            )
            for metric, value in date_metrics.items():
                if metric != "incorrect_dates_examples":
                    results["dates"][metric].append(value)
                else:
                    results["incorrect_dates_all"].update(value)
            
            # Add non-date field evaluation
            non_date_metrics = evaluate_non_date_fields(
                gold["structured_transcription"],
                pred_struct
            )
            for metric, value in non_date_metrics.items():
                if metric != "incorrect_values_examples":
                    results["non_date_fields"][metric].append(value)
                else:
                    results["incorrect_values_all"].update(value)
    
    # Calculate averages
    for eval_type in ["raw_transcription", "structured_fields"]:
        for metric, values in results[eval_type].items():
            results["overall"][f"{eval_type}_{metric}_avg"] = sum(values) / len(values) if values else 0
    
    # Calculate date metrics averages
    for metric, values in results["dates"].items():
        results["overall"][f"dates_{metric}_avg"] = sum(values) / len(values) if values else 0
    
    return results

if __name__ == "__main__":
    # Load data
    gold_data = load_gold_standard("madison_slates_annotation_omitted_removed/img_arr_prog.js")
    pred_data = load_predictions("llava_output_2")
    
    if gold_data and pred_data:
        # Run evaluation
        results = evaluate_predictions(gold_data, pred_data)
        
        # Print results
        print("\nDetailed Evaluation Report:")
        print("\n1. Raw Transcription Metrics:")
        print(f"  - Character Error Rate (CER): {results['overall']['raw_transcription_cer_avg']:.3f}")
        print(f"  - Word Error Rate (WER): {results['overall']['raw_transcription_wer_avg']:.3f}")
        
        print("\n2. Structured Fields Metrics (Exact field name and value matches):")
        print(f"  - Field-level Accuracy: {results['overall']['structured_fields_accuracy_avg']:.3f}")
        
        # Date metrics
        total_gold_dates = sum(results['dates']['total_gold_dates'])
        total_correct_dates = sum(results['dates']['correct_dates'])
        total_incorrect_dates = sum(results['dates']['incorrect_pred_dates'])
        
        print("\n3. Date Identification Metrics:")
        print("  These metrics evaluate date matching regardless of field names")
        print(f"  - Total dates in gold standard: {total_gold_dates}")
        print(f"  - Dates correctly identified: {total_correct_dates} ({(total_correct_dates/total_gold_dates)*100:.1f}% recall)")
        print(f"  - Incorrect dates in predictions: {total_incorrect_dates}")
        
        # Non-date field metrics
        total_gold_values = sum(results['non_date_fields']['total_gold_values'])
        total_correct_values = sum(results['non_date_fields']['correct_values'])
        total_pred_values = sum(results['non_date_fields']['total_pred_values'])
        total_incorrect_values = sum(results['non_date_fields']['incorrect_values'])
        
        print("\n4. Non-Date Field Metrics:")
        print("  These metrics evaluate value matching regardless of field names")
        print(f"  - Total values in gold standard: {total_gold_values}")
        print(f"  - Values correctly identified: {total_correct_values} ({(total_correct_values/total_gold_values)*100:.1f}% recall)")
        print(f"  - Total values in predictions: {total_pred_values}")
        print(f"  - Incorrect values in predictions: {total_incorrect_values}")
        print(f"  - Precision: {(total_correct_values/total_pred_values)*100:.1f}%")
        
        print("\n5. Sample of incorrect dates found in predictions:")
        incorrect_dates_sample = list(results["incorrect_dates_all"])[:10]
        for date in incorrect_dates_sample:
            print(f"  - {date}")
            
        print("\n6. Sample of incorrect non-date values found in predictions:")
        incorrect_values_sample = list(results["incorrect_values_all"])[:10]
        for value in incorrect_values_sample:
            print(f"  - {value}")
