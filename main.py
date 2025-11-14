# main.py
from training.preprocess import preprocess
from training.train import train_model

def main():
    print("🔍 Starting preprocessing...")
    x, y, tokenizer, max_len = preprocess()
    print("✅ Preprocessing complete!")
    print(f"Vocabulary size: {len(tokenizer.word_index)} | Max sequence length: {max_len}")

    print("\n🚀 Starting model training...")
    model = train_model(x, y, vocab_size=len(tokenizer.word_index) + 1, max_len=max_len)
    print("🎉 Training finished!")

if __name__ == "__main__":
    main()
