from utils import load_dataset, train_model  
def main():
    df = load_dataset("Dry_Bean_Dataset.csv")
    train_model(df)

if __name__ == "__main__":
    main()
