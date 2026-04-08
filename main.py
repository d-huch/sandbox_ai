from src.predict import predict_log
from src.train import train_model


def main():
    print("1 - Train model")
    print("2 - Predict log")
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
    else:
        print("Unknown option")


if __name__ == "__main__":
    main()