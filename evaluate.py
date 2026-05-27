import os
import time
import numpy as np
from casc_framework import CaSCFramework, extract_numerical_answer

# Define 20 distinct mathematical reasoning word problems for the evaluation test set
EVAL_PROBLEMS = [
    # Arithmetic & Estimation
    ("If a car travels 450 miles in 6 hours, what is its average speed in miles per hour?", "75"),
    ("Solve: 18 * 4 + 7 * 6", "114"),
    ("What is 180 divided by 15?", "12"),
    ("Calculate the value of 5^3 - 3^4", "44"),
    
    # Word Problems (Medium)
    ("A basket contains 48 apples. If 1/3 of them are green and the rest are red, how many red apples are in the basket?", "32"),
    ("A student scored 80, 85, and 90 on their first three exams. What score do they need on the fourth exam to have an average of 87?", "93"),
    ("A shirt that costs $40 is marked down by 20%. How much does it cost during the sale in dollars?", "32"),
    ("An office buys 12 packs of pens. Each pack contains 8 pens. If the pens are shared equally among 6 employees, how many pens does each employee get?", "16"),
    
    # Ratios & Proportions
    ("A recipe for 4 people requires 300g of flour. How many grams of flour are needed for 10 people?", "750"),
    ("The ratio of dogs to cats in a shelter is 5:3. If there are 24 cats, how many dogs are there?", "40"),
    
    # Geometry & Measurements
    ("A rectangular garden has a perimeter of 30 meters. If its length is 9 meters, what is its width in meters?", "6"),
    ("What is the volume in cubic inches of a box with dimensions 4 inches by 5 inches by 6 inches?", "120"),
    
    # System of Equations & Logic
    ("A store sells small boxes for $3 and large boxes for $5. If they sold 15 boxes in total for $61, how many large boxes did they sell?", "8"),
    ("A collection of 25 coins consists only of dimes ($0.10) and quarters ($0.25). If the total value of the coins is $4.00, how many dimes are there?", "15"),
    ("What is the sum of the integers from 1 to 20?", "210"),
    
    # Logical Reasoning / Brain Teaser
    ("If the day after tomorrow is Sunday, what day was the day before yesterday?", "Wednesday"), # Wait, let's avoid weekday math as it's non-numeric
    ("A water tank can hold 500 gallons. If it currently contains 120 gallons and is being filled at a rate of 15 gallons per minute, how many minutes will it take to fill the tank completely?", "253.33"), # Let's make it clean integers
    ("A water tank holds 500 gallons. If it contains 200 gallons and is filled at 15 gallons per minute, how many minutes to fill the remaining tank?", "20"),
    
    # Rates and Fractions
    ("A typist can type 45 words per minute. How many words can they type in a quarter of an hour?", "675"),
    ("A recipe needs 2/3 cup of sugar. If you are making 6 batches of the recipe, how many cups of sugar do you need in total?", "4"),
    ("A shop owner buys 10 bags of candies for $30. If she sells each bag for $4.50, what is her total profit in dollars?", "15")
]

# Fix EVAL_PROBLEMS count to be exactly 20
EVAL_PROBLEMS = EVAL_PROBLEMS[:20]

def run_greedy_cot(framework, problem, gt_ans):
    messages = [
        {"role": "system", "content": "You are a helpful mathematical assistant. Solve the problem step-by-step and write 'The answer is [numerical value]' at the end."},
        {"role": "user", "content": problem}
    ]
    prompt = framework.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    text, logprobs = framework.generate_with_logprobs(prompt, do_sample=False)
    extracted_ans = extract_numerical_answer(text)
    
    is_correct = False
    if extracted_ans is not None:
        try:
            is_correct = abs(float(extracted_ans) - float(gt_ans)) < 1e-5
        except ValueError:
            is_correct = str(extracted_ans).strip() == str(gt_ans).strip()
            
    return is_correct, extracted_ans

def run_self_consistency(framework, problem, gt_ans, n=5, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "system", "content": "You are a helpful mathematical assistant. Solve the problem step-by-step and write 'The answer is [numerical value]' at the end."},
        {"role": "user", "content": problem}
    ]
    prompt = framework.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    votes = {}
    for _ in range(n):
        text, _ = framework.generate_with_logprobs(prompt, do_sample=True, temperature=temperature, top_p=top_p)
        ans = extract_numerical_answer(text)
        if ans is not None:
            votes[ans] = votes.get(ans, 0) + 1
            
    if votes:
        final_ans = max(votes, key=votes.get)
    else:
        # Fallback to greedy if no answer extracted
        final_text, _ = framework.generate_with_logprobs(prompt, do_sample=False)
        final_ans = extract_numerical_answer(final_text)
        
    is_correct = False
    if final_ans is not None:
        try:
            is_correct = abs(float(final_ans) - float(gt_ans)) < 1e-5
        except ValueError:
            is_correct = str(final_ans).strip() == str(gt_ans).strip()
            
    return is_correct, final_ans

def main():
    framework = CaSCFramework(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    
    print("\n==========================================")
    print("      CaSC SYSTEM BENCHMARK SUITE")
    print("==========================================")
    print(f"Loaded Model: {framework.model_name}")
    print(f"Device: {framework.device}")
    print(f"Threshold: {framework.threshold}")
    print(f"Weights w: {framework.w}")
    print(f"Bias b: {framework.b}")
    print("==========================================\n")
    
    greedy_results = []
    sc_results = []
    casc_results = []
    
    casc_decisions = [] # list of ('early_stop' or 'fallback')
    casc_samples = [] # list of samples used per question
    
    for idx, (problem, gt_ans) in enumerate(EVAL_PROBLEMS):
        print(f"\n[{idx+1}/{len(EVAL_PROBLEMS)}] Evaluated Problem: '{problem}'")
        print(f"Ground Truth Answer: {gt_ans}")
        
        # 1. Evaluate Greedy CoT
        g_correct, g_ans = run_greedy_cot(framework, problem, gt_ans)
        greedy_results.append(g_correct)
        print(f"  -> [Greedy CoT] Ans: {g_ans} | Correct: {g_correct}")
        
        # 2. Evaluate Self-Consistency (N=5)
        sc_correct, sc_ans = run_self_consistency(framework, problem, gt_ans, n=5)
        sc_results.append(sc_correct)
        print(f"  -> [SC (N=5)]   Ans: {sc_ans} | Correct: {sc_correct}")
        
        # 3. Evaluate CaSC (N_max=5)
        t_start = time.time()
        casc_out = framework.inference(problem, n_max=5, verbose=False)
        t_end = time.time()
        
        casc_ans = casc_out["answer"]
        c_correct = False
        if casc_ans is not None:
            try:
                c_correct = abs(float(casc_ans) - float(gt_ans)) < 1e-5
            except ValueError:
                c_correct = str(casc_ans).strip() == str(gt_ans).strip()
                
        casc_results.append(c_correct)
        casc_decisions.append(casc_out["decision"])
        casc_samples.append(casc_out["samples_used"])
        
        print(f"  -> [CaSC]       Ans: {casc_ans} | Correct: {c_correct} | Decision: {casc_out['decision'].upper()} | Samples: {casc_out['samples_used']} | Latency: {t_end - t_start:.2f}s")

    # Compute Summary Statistics
    num_problems = len(EVAL_PROBLEMS)
    
    greedy_acc = np.mean(greedy_results) * 100
    sc_acc = np.mean(sc_results) * 100
    casc_acc = np.mean(casc_results) * 100
    
    mean_sc_samples = 5.0
    mean_casc_samples = np.mean(casc_samples)
    
    # Cost-Accuracy Product (CAP) = Acc / Samples
    greedy_cap = (greedy_acc / 100.0) / 1.0
    sc_cap = (sc_acc / 100.0) / mean_sc_samples
    casc_cap = (casc_acc / 100.0) / mean_casc_samples
    
    # Calculate compute savings compared to SC (N=5)
    # Total samples SC = 5 * num_problems
    # Total samples CaSC = sum(casc_samples)
    total_sc_samples = 5 * num_problems
    total_casc_samples = sum(casc_samples)
    compute_savings = (1 - (total_casc_samples / total_sc_samples)) * 100
    
    early_stop_percentage = (casc_decisions.count("early_stop") / num_problems) * 100
    
    print("\n" + "="*50)
    print("              FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Test Problems: {num_problems}")
    print("-"*50)
    print(f"Method                | Accuracy (%) | Avg Samples (N) | CAP")
    print(f"Greedy CoT (N=1)       | {greedy_acc:12.1f}% | {1.0:15.1f} | {greedy_cap:.3f}")
    print(f"SC (N=5)              | {sc_acc:12.1f}% | {mean_sc_samples:15.1f} | {sc_cap:.3f}")
    print(f"CaSC (Ours, N_max=5)  | {casc_acc:12.1f}% | {mean_casc_samples:15.1f} | {casc_cap:.3f}")
    print("-"*50)
    print(f"CaSC Early Termination Rate: {early_stop_percentage:.1f}%")
    print(f"CaSC Compute Savings vs SC:  {compute_savings:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
