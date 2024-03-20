import argparse
from .prepare_data import prepare_data
from .generate_models_A import generate_models_A
from .generate_models_B import generate_models_B

def main():

    """The main entry point of the application."""
    # In command prompt: python -m main --function xxx --year xxx --suffix xxx

    parser = argparse.ArgumentParser(description="Prepare DVF data, generate models or update streamlit for demo")

    # Add more arguments as needed
    parser.add_argument("--function", help="prepare_data, generate_model, setup_streamlit")
    parser.add_argument("--year", help="Indicate 'all' if all files in DVF directory must be taken. Or only the year you want if you want to process only one DVF file")
    parser.add_argument("--suffix", help="Indicate suffix that will be used for each temp file and for logs")
    parser.add_argument("--limit", help="Indicate limit to generate models B on a sample of classified ads")

    args = parser.parse_args()

    print(f"Starting application with arguments: {args}")

    # Application logic
    if args.function == 'prepare_data':
        try:
            prepare_data(args.year, args.suffix)
        except Exception as e:
            print(e)
    elif args.function == 'generate_models_A':
        try:
            generate_models_A(args.suffix)
        except Exception as e:
            print(e)
    elif args.function == 'generate_models_B':
        try:
            generate_models_B(int(args.limit))
        except Exception as e:
            print(e)
    else:
        print(f'Function {args.function} was not found in the code')


if __name__ == "__main__":
    main()
