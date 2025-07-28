# main.py
import argparse
import json
import os
from document_analyst import DocumentAnalyst

def main():
    """Main execution script for the Intelligent Document Analyst."""
    parser = argparse.ArgumentParser(description="Intelligent Document Analyst")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON configuration file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output JSON result.')
    parser.add_argument('--models', type=str, default='./models', help='Path to the directory containing pre-downloaded models.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        analyst = DocumentAnalyst(models_path=args.models)
        result = analyst.analyze(input_config_path=args.input)

        if result:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully saved analysis results to {args.output}")
        else:
            print("Analysis could not be completed.")

    except FileNotFoundError as e:
        print(f"Error: A required file or directory was not found: {e}")
        print("Please ensure the model and input file paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()