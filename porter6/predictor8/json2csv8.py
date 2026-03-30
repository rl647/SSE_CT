import json
import pandas as pd

# Define the mapping for the eight-state secondary structures
eight_state_classes_reverse = {0: 'G', 1: 'H', 2: 'I', 3: 'E', 4: 'B', 5: 'S', 6: 'T', 7: 'C'}

# Function to decode predicted classes into secondary structure symbols
def decode_predicted_classes(pred_string):
    return ''.join([eight_state_classes_reverse[int(char)] for char in pred_string])

# Function to convert probabilities to predicted class labels (0-7)
def convert_probabilities_to_labels(true_pred):
    return [str(max(range(len(prob)), key=lambda i: prob[i])) for prob in true_pred]

# Load JSON file and process
def process_json_to_csv(json_file_path, csv_output_path):
    # Load the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Prepare data for conversion
    records = []
    for record in json_data:
        if isinstance(record, dict) and 'id' in record and 'true_pred' in record:
            protein_id = record['id']
            true_pred = record['true_pred']
            # Convert probabilities to predicted class labels (0-7)
            predicted_classes = convert_probabilities_to_labels(true_pred)
            predicted_classes_str = ''.join(predicted_classes)
            # Decode the predicted classes to q8 symbols
            decoded_q8 = decode_predicted_classes(predicted_classes_str)
            records.append({
                'id': protein_id,
                'q8': decoded_q8
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)

# Example usage
json_file_path = '/home/people/20203023/scratch/porter6/predictor8/test_ensemble.json'  # Replace with your JSON file path
csv_output_path = '/home/people/20203023/scratch/porter6/predictor8/output_predictions.csv'  # Replace with your desired output CSV file path
process_json_to_csv(json_file_path, csv_output_path)

print(f"Processed data saved to {csv_output_path}")
