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
    # Remove ```json and ``` markers and any surrounding whitespace/newlines
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    text = re.sub(r'\s*`+\s*$', '', text)  # Remove any trailing backticks
    text = text.strip()
    
    try:
        return json.loads(text)
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
    for value in pred_dict.values():
        normalized_date = normalize_date(value)
        try:
            datetime.strptime(normalized_date, '%Y-%m-%d')
            pred_dates.add(normalized_date)
        except (ValueError, TypeError):
            continue

    return {
        "correct_dates": len(gold_dates & pred_dates),  # intersection
        "total_gold_dates": len(gold_dates),
        "incorrect_pred_dates": len(pred_dates - gold_dates)  # dates in pred but not in gold
    }

def evaluate_predictions(gold_data: Dict, pred_data: Dict) -> Dict:
    """
    Evaluate all predictions against gold standard
    """
    results = {
        "raw_transcription": defaultdict(list),
        "structured_fields": defaultdict(list),
        "dates": defaultdict(list),
        "overall": {}
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
            struct_metrics = evaluate_structured_fields(
                gold["structured_transcription"],
                pred_struct
            )
            for metric, value in struct_metrics.items():
                results["structured_fields"][metric].append(value)
            
            # Add date evaluation
            date_metrics = evaluate_dates(
                gold["structured_transcription"],
                pred_struct
            )
            for metric, value in date_metrics.items():
                results["dates"][metric].append(value)
    
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
        print("\nEvaluation Results:")
        print("\nRaw Transcription Metrics:")
        print(f"Average CER: {results['overall']['raw_transcription_cer_avg']:.3f}")
        print(f"Average WER: {results['overall']['raw_transcription_wer_avg']:.3f}")
        
        print("\nStructured Fields Metrics:")
        print(f"Average Accuracy: {results['overall']['structured_fields_accuracy_avg']:.3f}")
        
        print("\nDate Matching Metrics:")
        print(f"Average Correct Dates: {results['overall']['dates_correct_dates_avg']:.2f}")
        print(f"Average Gold Dates per Image: {results['overall']['dates_total_gold_dates_avg']:.2f}")
        print(f"Average Incorrect Predicted Dates: {results['overall']['dates_incorrect_pred_dates_avg']:.2f}")
