import re
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def extract_numerical_answer(text):
    """
    Extracts the final numerical answer from a chain-of-thought response.
    Looks for expressions like 'The answer is X', '#### X' (GSM8K standard), 
    or simply the last number in the text.
    """
    # Look for GSM8K style '#### X'
    gsm8k_match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if gsm8k_match:
        return gsm8k_match[-1]
        
    # Look for 'The answer is X'
    phrase_match = re.search(r"the answer is\s*:?\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if phrase_match:
        return phrase_match.group(1)
        
    # Fallback: extract the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None

class CaSCFramework:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", weights_path="casc_weights.json", threshold=0.75):
        self.model_name = model_name
        self.weights_path = weights_path
        self.threshold = threshold
        
        # Default Logistic Regression Weights (trained on synthetic or initialized to reasonable values)
        # Feature order: [mean_conf, std_conf, len_norm_logprob, entropy_proxy]
        self.w = np.array([2.0, -1.0, 1.5, -0.5])
        self.b = 0.0
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing CaSC Framework on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Load weights if they exist
        self.load_weights()
        
    def save_weights(self):
        data = {
            "w": self.w.tolist(),
            "b": float(self.b),
            "threshold": float(self.threshold)
        }
        with open(self.weights_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved CaSC weights to {self.weights_path}")
        
    def load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r") as f:
                    data = json.load(f)
                self.w = np.array(data["w"])
                self.b = data["b"]
                self.threshold = data.get("threshold", self.threshold)
                print(f"Loaded CaSC weights from {self.weights_path}")
            except Exception as e:
                print(f"Error loading weights from {self.weights_path}: {e}. Using defaults.")

    def compute_features(self, logprobs):
        """
        Computes the 4 trajectory features from token logprobs:
        1. Mean token confidence
        2. Standard deviation of token confidence
        3. Length-normalized log-likelihood
        4. Entropy proxy
        """
        if not logprobs:
            return np.zeros(4)
        probs = np.exp(logprobs)
        mean_conf = np.mean(probs)
        std_conf = np.std(probs)
        len_norm_logprob = np.mean(logprobs)
        entropy_proxy = -np.mean(np.array(logprobs) * probs)
        return np.array([mean_conf, std_conf, len_norm_logprob, entropy_proxy])

    def predict_confidence(self, features):
        """
        Calculates p_hat (predicted confidence score) using logistic regression.
        """
        logit = np.dot(self.w, features) + self.b
        return sigmoid(logit)

    def generate_with_logprobs(self, prompt, do_sample=False, temperature=0.7, top_p=0.9, max_new_tokens=256):
        """
        Generates text and extracts token-level log probabilities.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[-1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        gen_tokens = outputs.sequences[0, input_len:]
        scores = outputs.scores
        
        logprobs = []
        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        for i in range(len(scores)):
            if i >= len(gen_tokens):
                break
            token_id = gen_tokens[i].item()
            logits = scores[i][0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)
            
        return generated_text, logprobs

    def inference(self, problem, n_max=5, temperature=0.7, top_p=0.9, verbose=False):
        """
        Executes the main CaSC decoding algorithm:
        1. Greedy decode first (y_0)
        2. Predict confidence (p_hat_0)
        3. Early stop if p_hat_0 >= threshold
        4. Otherwise fallback to sampling N_max additional paths and perform weighted voting
        """
        # Formulate instruction prompt for Qwen Instruct
        messages = [
            {"role": "system", "content": "You are a helpful mathematical assistant. Solve the problem step-by-step and write 'The answer is [numerical value]' at the end."},
            {"role": "user", "content": problem}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Step 1: Greedy Decode
        greedy_text, greedy_logprobs = self.generate_with_logprobs(prompt, do_sample=False)
        greedy_features = self.compute_features(greedy_logprobs)
        p_hat_0 = self.predict_confidence(greedy_features)
        greedy_ans = extract_numerical_answer(greedy_text)
        
        if verbose:
            print(f"\n--- Problem: {problem} ---")
            print(f"Greedy Output: {greedy_text}")
            print(f"Features: Mean={greedy_features[0]:.4f}, Std={greedy_features[1]:.4f}, NormLL={greedy_features[2]:.4f}, Entropy={greedy_features[3]:.4f}")
            print(f"Predicted Confidence (p_hat_0): {p_hat_0:.4f} (Threshold: {self.threshold})")
            print(f"Extracted Greedy Answer: {greedy_ans}")
            
        # Decision: Early Stop
        if p_hat_0 >= self.threshold:
            if verbose:
                print(">>> Action: [EARLY TERMINATION] (Greedy accepted)")
            return {
                "decision": "early_stop",
                "answer": greedy_ans,
                "text": greedy_text,
                "samples_used": 1,
                "confidence": p_hat_0,
                "all_samples": [(greedy_ans, p_hat_0, greedy_text)]
            }
            
        # Step 2: Fallback Sampling
        if verbose:
            print(f">>> Action: [FALLBACK SAMPLING] (Confidence {p_hat_0:.4f} < {self.threshold})")
            
        votes = {} # answer -> sum of confidence weights
        all_samples = []
        
        # Include greedy path in the vote set as well
        if greedy_ans is not None:
            votes[greedy_ans] = p_hat_0
            all_samples.append((greedy_ans, p_hat_0, greedy_text))
            
        for k in range(n_max):
            sampled_text, sampled_logprobs = self.generate_with_logprobs(
                prompt, do_sample=True, temperature=temperature, top_p=top_p
            )
            sampled_features = self.compute_features(sampled_logprobs)
            p_hat_k = self.predict_confidence(sampled_features)
            sampled_ans = extract_numerical_answer(sampled_text)
            
            if verbose:
                print(f"  Sample {k+1}: Ans={sampled_ans}, Conf={p_hat_k:.4f} | Text: {sampled_text[:100]}...")
                
            if sampled_ans is not None:
                votes[sampled_ans] = votes.get(sampled_ans, 0.0) + p_hat_k
                all_samples.append((sampled_ans, p_hat_k, sampled_text))
                
        # Aggregate weighted votes
        if votes:
            final_ans = max(votes, key=votes.get)
            # Find the sample text associated with this final answer that has the highest confidence
            matching_samples = [s for s in all_samples if s[0] == final_ans]
            best_sample_text = max(matching_samples, key=lambda x: x[1])[2] if matching_samples else greedy_text
        else:
            final_ans = greedy_ans
            best_sample_text = greedy_text
            
        if verbose:
            print(f"Weighted Majority Vote Result: {final_ans} | Active Votes: {votes}")
            
        return {
            "decision": "fallback",
            "answer": final_ans,
            "text": best_sample_text,
            "samples_used": 1 + n_max,
            "confidence": votes.get(final_ans, 0.0) if votes else 0.0,
            "all_samples": all_samples
        }
