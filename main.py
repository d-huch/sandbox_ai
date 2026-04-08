from src.predict import predict_file, predict_log, resolve_input_path
from src.train import train_model


def main():
    print("1 - Train model")
    print("2 - Predict single log")
    print("3 - Predict from txt/csv file")
    choice = input("Choose action: ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        while True:
            text = input("\nEnter log (or 'exit'): ").strip()
            if text.lower() == "exit":
                print("Bye.")
                break

            label, score, normalized = predict_log(text)
            print(f"Normalized: {normalized}")
            print(f"Score: {score:.4f}")
            print(f"Prediction: {label}")
    elif choice == "3":
        path = input("Enter path to .txt or .csv file: ").strip()
        resolved_path = resolve_input_path(path)

        try:
            results = predict_file(path)
        except FileNotFoundError as e:
            if not resolved_path.is_file():
                print(f"File not found: {resolved_path}")
            else:
                print(f"Missing required file: {e}")
            return
        except ValueError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Prediction failed: {e}")
            return

        if not results:
            print("No valid rows found for prediction.")
            return

        for index, item in enumerate(results, start=1):
            print(f"\n[{index}] {item['text']}")
            print(f"Normalized: {item['normalized']}")
            print(f"Score: {item['score']:.4f}")
            print(f"Prediction: {item['prediction']}")
    else:
        print("Unknown option")


if __name__ == "__main__":
    main()
