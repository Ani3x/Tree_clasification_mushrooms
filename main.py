import pandas as pd
from tree import Node, gini_index, best_feature, build_tree, predict


# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("mushrooms.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}\n")


# ── 2. Train / test split (80 / 20, no external libraries needed) ─────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx].reset_index(drop=True)
test_df   = df.iloc[split_idx:].reset_index(drop=True)

print(f"Training samples : {len(train_df)}")
print(f"Testing  samples : {len(test_df)}\n")


# ── 3. Build the decision tree ────────────────────────────────────────────────
features = [col for col in df.columns if col != "class"]

print("Building decision tree …")
tree = build_tree(train_df, features, depth=0, max_depth=10)
print("Tree built successfully.\n")


# ── 4. Evaluate on the test set ───────────────────────────────────────────────
def evaluate(tree, test_df):
    correct = 0
    total   = len(test_df)
    unknown = 0

    for _, row in test_df.iterrows():
        prediction = predict(tree, row)

        if prediction is None:
            unknown += 1          # unseen feature value during training
            continue

        if prediction == row["class"]:
            correct += 1

    accuracy = correct / (total - unknown) if (total - unknown) > 0 else 0
    return accuracy, correct, total, unknown


accuracy, correct, total, unknown = evaluate(tree, test_df)

print("── Results ──────────────────────────────────────────────────────────────")
print(f"Correct predictions : {correct} / {total - unknown}")
print(f"Unpredictable rows  : {unknown}  (unseen feature values)")
print(f"Accuracy            : {accuracy * 100:.2f}%\n")


# ── 5. Show a few sample predictions ─────────────────────────────────────────
print("── Sample predictions (first 5 test rows) ───────────────────────────────")
print(f"{'#':<4} {'Actual':<10} {'Predicted':<10} {'Correct?'}")
print("-" * 36)

for i, (_, row) in enumerate(test_df.head(5).iterrows()):
    actual    = row["class"]
    predicted = predict(tree, row)
    mark      = "✓" if predicted == actual else "✗"
    print(f"{i+1:<4} {'edible' if actual == 'e' else 'poisonous':<10} "
          f"{'edible' if predicted == 'e' else 'poisonous':<10} {mark}")