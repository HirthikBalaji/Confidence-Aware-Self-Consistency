import os
import json
import numpy as np
from casc_framework import CaSCFramework, extract_numerical_answer

# Define a diverse dataset of 30 mathematical reasoning word problems
# This contains a mix of easy, medium, and slightly trickier problems to elicit both correct and incorrect answers.
TRAINING_PROBLEMS = [
    # Easy Arithmetic
    ("What is 47 + 58?", "105"),
    ("Solve: 15 * 6 - 20", "70"),
    ("What is 144 divided by 12?", "12"),
    ("Calculate: 25 * 4 + 13", "113"),
    ("What is the square root of 64?", "8"),
    
    # Word Problems (Easy)
    ("A pen costs $3. How much do 7 pens cost?", "21"),
    ("If John has 15 apples and gives 4 to Mary and 3 to Bob, how many apples does he have left?", "8"),
    ("A book has 120 pages. If Sarah reads 15 pages a day, how many days will it take her to finish the book?", "8"),
    ("A train travels at 60 miles per hour. How far will it travel in 3.5 hours?", "210"),
    ("If a rectangle has a length of 8 cm and a width of 5 cm, what is its area in square cm?", "40"),
    
    # Percentage & Ratios
    ("What is 15% of 200?", "30"),
    ("If a jacket originally costs $80 and has a 25% discount, what is the sale price?", "60"),
    ("In a class of 30 students, 60% are girls. How many boys are in the class?", "12"),
    ("Divide 50 in the ratio 2:3. What is the value of the larger share?", "30"),
    
    # Medium Algebraic/Reasoning Problems
    ("Solve for x: 3x + 7 = 22", "5"),
    ("Solve for y: 2y - 10 = 4y - 18", "4"),
    ("A father is 4 times as old as his son. In 5 years, the father will be 3 times as old as his son. How old is the son now?", "10"),
    ("If 5 machines can make 5 widgets in 5 minutes, how many minutes does it take 100 machines to make 100 widgets?", "5"),
    
    # Number Theory / Logic
    ("What is the sum of all prime numbers between 1 and 10?", "17"),
    ("A farmer has chickens and rabbits. If there are 10 heads and 32 legs, how many rabbits does he have?", "6"),
    ("What is the next number in the sequence: 2, 4, 8, 16, ...?", "32"),
    ("What is the next number in the sequence: 1, 1, 2, 3, 5, 8, ...?", "13"),
    
    # Multi-Step Word Problems
    ("A bakery sells cupcakes for $2 each. If you buy 6, you get a 10% discount on the total. How much do 6 cupcakes cost in dollars?", "10.8"),
    ("Tom has twice as many marbles as Jerry. If Tom gives Jerry 5 marbles, they will have the same number. How many marbles did Tom start with?", "20"),
    ("A swimming pool is being filled with water. The pipe fills it at 50 gallons per minute, but there is a leak discharging 10 gallons per minute. What is the net rate of filling in gallons per minute?", "40"),
    
    # Challenging Arithmetic/Fractions
    ("Solve: 1/2 + 1/4 + 1/8", "0.875"),
    ("What is 3/5 of 150?", "90"),
    ("A car rental costs $30 a day plus $0.15 per mile. If you rent a car for 2 days and drive 100 miles, what is the total cost in dollars?", "75"),
    ("What is the least common multiple of 6 and 8?", "24"),
    ("If x = 3 and y = 4, what is the value of x^2 + y^2?", "25")
]

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def collect_training_data(framework):
    """
    Runs the model in greedy decode on training problems, records features,
    and checks if the answer matches the ground truth to assign a 0 or 1 label.
    """
    features_list = []
    labels_list = []
    
    print("\n=== STEP 1: COLLECTING CALIBRATION DATA ===")
    
    for idx, (problem, gt_ans) in enumerate(TRAINING_PROBLEMS):
        print(f"\nProcessing Problem {idx+1}/{len(TRAINING_PROBLEMS)}:")
        print(f"Problem: {problem}")
        print(f"Ground Truth Answer: {gt_ans}")
        
        # Build instruction prompt
        messages = [
            {"role": "system", "content": "You are a helpful mathematical assistant. Solve the problem step-by-step and write 'The answer is [numerical value]' at the end."},
            {"role": "user", "content": problem}
        ]
        prompt = framework.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            # Generate greedy path and logprobs
            greedy_text, logprobs = framework.generate_with_logprobs(prompt, do_sample=False)
            features = framework.compute_features(logprobs)
            extracted_ans = extract_numerical_answer(greedy_text)
            
            # Label = 1 if correctly answered, 0 if incorrect
            is_correct = False
            if extracted_ans is not None:
                try:
                    # Parse as floats to handle things like "5" vs "5.0"
                    is_correct = abs(float(extracted_ans) - float(gt_ans)) < 1e-5
                except ValueError:
                    is_correct = str(extracted_ans).strip() == str(gt_ans).strip()
            
            label = 1.0 if is_correct else 0.0
            
            print(f"Model Output: {greedy_text.strip()}")
            print(f"Extracted: '{extracted_ans}' | Correct: {is_correct}")
            print(f"Features: Mean={features[0]:.4f}, Std={features[1]:.4f}, NormLL={features[2]:.4f}, Entropy={features[3]:.4f}")
            
            features_list.append(features)
            labels_list.append(label)
            
        except Exception as e:
            print(f"Error processing problem {idx+1}: {e}")
            
    return np.array(features_list), np.array(labels_list)

def train_logistic_regression(X, y, epochs=2000, lr=0.1, lambda_reg=0.01):
    """
    Trains a logistic regression model with L2 regularization using gradient descent.
    """
    print("\n=== STEP 2: TRAINING LOGISTIC REGRESSION GATE ===")
    N, D = X.shape
    
    # Initialize weights to reasonable values to start
    # Feature indices: 0: mean_conf, 1: std_conf, 2: len_norm_logprob, 3: entropy_proxy
    w = np.zeros(D)
    b = 0.0
    
    # Let's verify class distribution
    num_correct = int(np.sum(y))
    num_incorrect = N - num_correct
    print(f"Training set sizes: {N} samples. Class distribution: Correct={num_correct}, Incorrect={num_incorrect}")
    
    # If the training set is completely skewed (e.g., model got all correct), we insert synthetic prior boundary samples
    # to guarantee stable convergence of decision boundaries.
    if num_correct == N or num_incorrect == N:
        print("Warning: Single-class training data observed. Augmenting training set with prior boundary anchors.")
        # Anchor correct sample features: high confidence, low entropy
        anchor_correct = np.array([0.95, 0.05, -0.05, 0.02])
        # Anchor incorrect sample features: low confidence, high entropy
        anchor_incorrect = np.array([0.40, 0.30, -0.90, 0.36])
        
        X = np.vstack([X, anchor_correct, anchor_incorrect])
        y = np.append(y, [1.0, 0.0])
        N, D = X.shape
    
    for epoch in range(epochs):
        logits = np.dot(X, w) + b
        predictions = sigmoid(logits)
        
        epsilon = 1e-15
        loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        loss += (lambda_reg / (2 * N)) * np.sum(w ** 2) # L2 Regularization
        
        dw = (1 / N) * np.dot(X.T, (predictions - y)) + (lambda_reg / N) * w
        db = (1 / N) * np.sum(predictions - y)
        
        w -= lr * dw
        b -= lr * db
        
        if (epoch + 1) % 400 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")
            
    print(f"Final Trained weights: {w}")
    print(f"Final Trained bias: {b:.6f}")
    return w, b

def main():
    framework = CaSCFramework(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    
    X, y = collect_training_data(framework)
    
    if len(X) == 0:
        print("Error: No data was collected. Aborting training.")
        return
        
    w, b = train_logistic_regression(X, y)
    
    # Update framework and save weights
    framework.w = w
    framework.b = b
    framework.save_weights()
    
    print("\nGate training completed successfully!")

if __name__ == "__main__":
    main()
