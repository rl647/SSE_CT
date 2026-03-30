import json
import pandas as pd
import sys
p1 = sys.argv[2]
# Define the mapping for the three-state secondary structures
q3_decoding = {0: 'H', 1: 'E', 2: 'C'}

# Function to convert probabilities to predicted class labels (0-2)
def convert_probabilities_to_q3_labels(true_pred):
    return [str(max(range(len(prob)), key=lambda i: prob[i])) for prob in true_pred]

# Function to decode predicted q3 classes into secondary structure symbols
def decode_q3(pred_string):
    return ''.join([q3_decoding[int(char)] for char in pred_string])

# Load JSON file and process
def process_json_to_q3_csv(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

   

    # Prepare data for conversion
    with open(f'{p1}','a') as f:
        for record in json_data:
            if isinstance(record, dict) and 'id' in record and 'true_pred' in record:
                protein_id = record['id']
                true_pred = record['true_pred']
                # Convert probabilities to predicted class labels (q3 with values 0, 1, 2)
                predicted_classes = convert_probabilities_to_q3_labels(true_pred)
                predicted_classes_str = ''.join(predicted_classes)
                # Decode q3 labels into secondary structure symbols (H, E, C)
                decoded_q3 = decode_q3(predicted_classes_str)
                f.write(protein_id+'\t'+decoded_q3+'\n')
    #             records.append({
    #                 'id': protein_id,
    #                 'q3_decoded': decoded_q3
    #             })
    # # Create DataFrame and save to CSV
    # df = pd.DataFrame(records)
    # df.to_csv(csv_output_path, index=False)

# Example usage
json_file_path = sys.argv[1]  # Replace with your JSON file path
process_json_to_q3_csv(json_file_path)

print(f"Processed data saved to ")
