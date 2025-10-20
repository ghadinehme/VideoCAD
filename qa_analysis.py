from eval_qa import QADataset, Evaluator, ModelOutput
from typing import List
import json, random
import os
from collections import Counter

def load_model_outputs(model_outputs_path: str, qa_dataset: QADataset) -> List[ModelOutput]:
    with open(model_outputs_path, 'r') as f:
        model_outputs = json.load(f)
    for output in model_outputs:
        qa_item = qa_dataset.get_item_by_id(output["question_id"])
        output["template"] = qa_item.template
    return model_outputs

def load_all_model_outputs(model_outputs_path: str, qa_dataset: QADataset) -> List[ModelOutput]:
    model_outputs = []
    for file in os.listdir(model_outputs_path):
        if "results" not in file:
            continue
        outputs = load_model_outputs(os.path.join(model_outputs_path, file), qa_dataset)
        model_outputs.extend(outputs)
    for model_output in model_outputs:
        if model_output.get("answer") != "unknown_answer_code":
            continue
        # Replace "unknown_answer_code" with a random option
        qa_item = qa_dataset.get_item_by_id(model_output["question_id"])
        option_ids = [opt.id for opt in qa_item.options]
        # Replace with a random option ID
        model_output["answer"] = random.choice(option_ids)
        print(f"Replaced unknown_answer_code with random option {model_output['answer']} for question {model_output['question_id']}")
    
    return model_outputs

def sample_model_outputs(model_outputs: List[ModelOutput], sample_size: int = 200) -> List[ModelOutput]:
    outputs_by_template = {}
    # First filter out unknown answers
    valid_outputs = [output for output in model_outputs if output["answer"] != "unknown_answer_code"]
    
    for output in valid_outputs:
        template = output["template"]
        if template not in outputs_by_template:
            outputs_by_template[template] = []
        outputs_by_template[template].append(output)

    # Sample up to 200 outputs from each template
    sampled_outputs = []
    for template, outputs in outputs_by_template.items():
        sample_size = min(200, len(outputs))
        if len(outputs) < 200:
            raise ValueError(f"Expected 200 outputs for template {template}, got {len(outputs)}")
        sampled_outputs.extend(random.sample(outputs, sample_size))
        print(f"Sampled {sample_size} outputs from {template}")

    print(f"\nTotal sampled outputs: {len(sampled_outputs)}")
    return sampled_outputs

def group_outputs(model_outputs: List[ModelOutput], start_q: int, end_q: int):
    outputs_by_question_id = {}
    for output in model_outputs:
        q_id = output["question_id"]
        if not (start_q <= q_id <= end_q):
            continue
        if q_id not in outputs_by_question_id:
            outputs_by_question_id[q_id] = []
        outputs_by_question_id[q_id].append(output)

    valid_groups = {}
    invalid_groups = []
    for q_id, outputs in outputs_by_question_id.items():
        if len(outputs) < 3:
            invalid_groups.append(q_id)
            print(f"Warning: Question ID {q_id} has less than 3 samples (has {len(outputs)}). Skipping.")

        if len(outputs) % 2 == 0:
            if len(outputs) == 2:
                invalid_groups.append(q_id)
                # print(f"Warning: Question ID {q_id} has an even number of samples (has {len(outputs)}). Skipping.")
                continue
            # Find the least represented answer or randomly choose if tied
            answer_counts = Counter(output["answer"] for output in outputs)
            min_count = min(answer_counts.values())
            least_common = [ans for ans, count in answer_counts.items() if count == min_count]
            answer_to_remove = random.choice(least_common)
            
            # Remove one instance of the chosen answer
            for i, output in enumerate(outputs):
                if output["answer"] == answer_to_remove:
                    outputs.pop(i)
                    break
            # print(f"Warning: Question ID {q_id} has an even number of samples (has {len(outputs)}). Skipping.")

        valid_groups[q_id] = outputs
    
    if not valid_groups:
        print("No valid groups found after filtering. Exiting.")
        return None # Or raise an exception
    if invalid_groups:
        invalid_groups = sorted(list(set(invalid_groups)))
        print(f"Invalid groups: {invalid_groups}")
        # raise Exception("Invalid groups found after filtering. Exiting.")
    # Print template counts before returning
    print("\nTemplate counts in valid groups:")
    template_counts = {}
    for q_id, outputs in valid_groups.items():
        template = outputs[0]["template"]  # All outputs for same q_id have same template
        template_counts[template] = template_counts.get(template, 0) + 1
    
    for template, count in template_counts.items():
        print(f"  {template}: {count} questions")

    return valid_groups

def evaluate_outputs(qa_dataset: QADataset, model_outputs: List[ModelOutput], start_q, end_q):
    """
    Evaluates the model outputs and returns a dictionary of results.
    """
    grouped_outputs = group_outputs(model_outputs, start_q, end_q)
    if not grouped_outputs:
        return {}

    results_by_template = {}
    for q_id, outputs in grouped_outputs.items():
        # All outputs for a given q_id should have the same template
        # and the same ground truth answer from the qa_dataset.
        qa_item = qa_dataset.get_item_by_id(q_id)
        template_name = qa_item.template

        if template_name not in results_by_template:
            results_by_template[template_name] = {"correct": 0, "total": 0}

        answers = [output['answer'] for output in outputs]
        answer_counts = Counter(answers)
        mode_answer, _ = answer_counts.most_common(1)[0]
        
        is_question_correct = (mode_answer == qa_item.answer) # Compare mode to ground truth

        results_by_template[template_name]["total"] += 1
        if is_question_correct:
            results_by_template[template_name]["correct"] += 1
            
    # Tally and print results
    print("\nEvaluation Results:")
    overall_correct = 0
    overall_total = 0
    for template_name, counts in results_by_template.items():
        accuracy = (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        print(f"  Template: {template_name}")
        print(f"    Correct: {counts['correct']}")
        print(f"    Total: {counts['total']}")
        print(f"    Accuracy: {accuracy:.2f}%")
        overall_correct += counts['correct']
        overall_total += counts['total']
    
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print("\n  Overall Performance:")
    print(f"    Total Correct: {overall_correct}")
    print(f"    Total Questions: {overall_total}")
    print(f"    Overall Accuracy: {overall_accuracy:.2f}%")
    # Create final results dictionary with overall metrics
    final_results = {
        template: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"]/counts["total"]
        }
        for template, counts in results_by_template.items()
    }

    # Add overall metrics
    final_results["overall_accuracy"] = overall_accuracy
    final_results["num_questions"] = overall_total
    final_results["num_correct"] = overall_correct

    return final_results


def evaluate_path(qa_dataset: QADataset, evaluator: Evaluator, model_outputs_path: str):
    model_outputs = load_model_outputs(model_outputs_path, qa_dataset)
    sampled_outputs = sample_model_outputs(model_outputs)
    evaluator.evaluate(sampled_outputs)

def evaluate_path_by_template(qa_dataset: QADataset, evaluator: Evaluator, model_outputs_path: str):
    model_outputs = load_all_model_outputs(model_outputs_path, qa_dataset)
    results = evaluate_outputs(qa_dataset, model_outputs, 0, 1200)
    return results


def eval_model(model_output_path: str, qa_dataset: QADataset, evaluator: Evaluator):
    results = evaluate_path_by_template(qa_dataset, evaluator, 
                                        model_output_path)
    results["model"] = model_output_path.split("/")[-1]
    print(results)

    # Load existing results
    existing_results = []
    json_path = "res_final.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            except json.JSONDecodeError:
                existing_results = []
    
    # Append new results
    existing_results.append(results)
    
    # Write back all results
    with open(json_path, "w") as f:
        json.dump(existing_results, f, indent=4)

paths = [
"data/data_raw/qa/results/gemini-2.5-pro-preview-05-06",
"data/data_raw/qa/results/gpt-4.1-2025-04-14",
"data/data_raw/qa/results/o3-2025-04-16",
"data/data_raw/qa/results/claude-3-7-sonnet-20250219",
"data/data_raw/qa/results/qwen2.5-vl-72b-instruct"

]

def main():
    # model_outputs_path = "data/data_raw/qa/results/random/model_results.json"
    model_output_path = "data/data_raw/qa/results/gemini-2.5-pro-preview-05-06"
    qa_dataset = QADataset("data/data_raw/qa.json")
    evaluator = Evaluator(qa_dataset)

    for path in paths:
        eval_model(path, qa_dataset, evaluator)


    

if __name__ == "__main__":
    main()