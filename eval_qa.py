import argparse
import json
import os
import base64 # Added for image encoding
from typing import List, Dict, Literal, Optional, Any, Union
from pathlib import Path
from collections import Counter # Added for majority vote

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import dotenv
import re
import time
import random
dotenv.load_dotenv()

# --- Configuration ---
# TODO: Set your OpenAI API Key here or ensure it is in your environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
DEFAULT_MODEL_NAME = "gpt-4.1-2025-04-14" # Changed to a vision model

# --- Pydantic Models for Data Validation ---

class QuestionDetail(BaseModel):
    text: str
    image_refs: Optional[dict] = {}

class Option(BaseModel):
    id: str
    text: Optional[Union[str, int, float]] = None # This can be actual text or a path to an image for image options
    path: Optional[str] = None # This can be actual text or a path to an image for image options
    # image_ref: Optional[str] = None # Add this if you prefer an explicit field for option images

class QAItem(BaseModel):
    question: QuestionDetail
    options: List[Option]
    answer: str  # This is the ID of the correct option
    template: str
    id: int

class ModelOutput(BaseModel):
    model: str
    answer: str  # The ID of the option chosen by the model
    question_id: int
    correct: Literal[0, 1]
    model_reasoning: Optional[str] = None

class TemplateAccuracy(BaseModel):
    pass # Will be dynamically created

class OverallResults(BaseModel):
    template_accuracies: Dict[str, float]
    overall_accuracy: float


UNKNOWN_ANSWER_CODE = "unknown_answer_code"

class LLMClientFactory:

    def __init__(self, model_name: str, api_key: Optional[str] = None, dataset_base_path: Optional[Path] = None, mock_mode: bool = True):
        self.model_name = model_name
        self.api_key = api_key
        self.dataset_base_path = dataset_base_path
        self.mock_mode = mock_mode

        self.open_ai_model_names = ["gpt", "o3", "openai"]
        self.anthropic_model_names = ["claude", "anthropic"]
        self.google_model_names = ["gemini", "google"]
        self.qwen_model_names = ["qwen", "qwen-max"]
        self.glm_model_names = ["glm", "glm-4v-plus"]
        self.opengvlab_model_names = ["internvl3-2b", "opengvlab"]
        self.random_model_names = ["random"]
    def identify_provider(self, model_name: str) -> str:    
        if any(model_name.startswith(prefix) for prefix in self.open_ai_model_names):
            return "openai"
        elif any(model_name.startswith(prefix) for prefix in self.anthropic_model_names):
            return "anthropic"
        elif any(model_name.startswith(prefix) for prefix in self.google_model_names):
            return "google"
        elif any(model_name.startswith(prefix) for prefix in self.qwen_model_names):
            return "qwen"
        elif any(model_name.startswith(prefix) for prefix in self.glm_model_names):
            return "glm"
        elif any(model_name.startswith(prefix) for prefix in self.opengvlab_model_names):
            return "opengvlab"
        elif any(model_name.startswith(prefix) for prefix in self.random_model_names):
            return "random"
        else:
            raise ValueError(f"Unsupported model: {model_name}. Please use a model from OpenAI, Anthropic, Google Gemini, or Qwen.")

    def get_llm_client(self, model_name: Union[str, tuple], api_key: Optional[str] = None):
        """
        Factory method to create the appropriate LLM client based on the model name.
        """
        if type(model_name) == tuple:
            name, provider = model_name
        else:
            name = model_name
            provider = self.identify_provider(model_name)
        # Determine which provider to use based on model name
        if provider == "openai":
            return self._create_openai_client(name, api_key)
        elif provider == "anthropic":
            return self._create_anthropic_client(name, api_key)
        elif provider == "google":
            return self._create_gemini_client(name, api_key)
        elif provider == "qwen":
            return self._create_qwen_client(name, api_key)
        elif provider == "glm":
            return self._create_glm_client(name, api_key)
        elif provider == "opengvlab":
            return self._create_opengvlab_client(name, api_key)
        elif provider == "random":
            return self._create_random()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Please use a model from OpenAI, Anthropic, Google Gemini, or Qwen.")
    
    def _create_qwen_client(self, model_name: str, api_key: Optional[str] = None):
        """Create a Qwen client."""
        if api_key:
            os.environ["QWEN_API_KEY"] = api_key
        if not os.getenv("QWEN_API_KEY"):
            raise ValueError("QWEN_API_KEY environment variable not set.")
        return ChatOpenAI(
                openai_api_key=os.getenv("QWEN_API_KEY"),
                openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                model_name=model_name,
                max_retries=3 # Added retry
            )
    
    def _create_opengvlab_client(self, model_name: str, api_key: Optional[str] = None):
        """Create a OpenGVLab client."""
        if api_key:
            os.environ["OPEN_ROUTER_KEY"] = api_key
        if not os.getenv("OPEN_ROUTER_KEY"):
            raise ValueError("OPEN_ROUTER_KEY environment variable not set.")
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.getenv("OPEN_ROUTER_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            max_retries=3 # Added retry
        )
    
    def _create_glm_client(self, model_name: str, api_key: Optional[str] = None):
        """Create a GLM client."""
        if api_key:
            os.environ["GLM_API_KEY"] = api_key
        if not os.getenv("GLM_API_KEY"):
            raise ValueError("GLM_API_KEY environment variable not set.")
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
            max_retries=3 # Added retry
        )
    
    def _create_openai_client(self, model_name: str, api_key: Optional[str] = None):
        """Create an OpenAI client."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return ChatOpenAI(
             model_name=model_name,
             max_retries=3 # Added retry
        )
    
    def _create_random(self):
        return "random"
    
    def _create_anthropic_client(self, model_name: str, api_key: Optional[str] = None):
        """Create an Anthropic client."""
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        
        return ChatAnthropic(model_name=model_name, temperature=0, max_retries=3) # Added retry
    
    def _create_gemini_client(self, model_name: str, api_key: Optional[str] = None):
        """Create a Google Gemini client."""
        if not api_key:
            # Try to get from environment if not directly provided
            api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key: # Check after trying to get from env
            raise ValueError("GOOGLE_API_KEY not provided directly or as environment variable.")
        return ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=api_key,
            temperature=0,
            max_retries=3 # Added retry
        )

class LLMClient:
    """
    Client to interact with the specified Large Language Model.
    Supports OpenAI, Anthropic, and Google Gemini models.
    """
    def __init__(self, 
                 model_name: Union[str, tuple] = DEFAULT_MODEL_NAME, 
                 api_key: Optional[str] = None, 
                 dataset_base_path: Optional[Path] = None,
                 mock_mode: bool = False,
                 random_mode: bool = False, # User added this
                 request_delay_seconds: float = 1.0): # Added request_delay_seconds
        self.llm_factory = LLMClientFactory(model_name, api_key, dataset_base_path, mock_mode)
        self.llm = self.llm_factory.get_llm_client(model_name, api_key)
        self.mock_mode = mock_mode
        self.random_mode = random_mode # User added this
        self.request_delay_seconds = request_delay_seconds # Store delay

        if type(model_name) == tuple:
            self.model_name = model_name[0]
        else:
            self.model_name = model_name

    def _image_to_base64(self, image_path: Path) -> Optional[str]:
        """Reads an image file and returns its base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _format_image_message(self, image_ref: str) -> Dict[str, Any]:
        """Formats an image reference into a message for the LLM."""
        img_path =  image_ref
        img_base64 = self._image_to_base64(img_path)
        return {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{img_base64}", "detail": "high"}}

        
    def _parse_question(self, qa_item: QAItem) -> str:
        """Parses the question text and handles placeholders for images."""
        question_text = qa_item.question.text
        # Use regex to replace any text within angle brackets with empty string
        question_text = re.sub(r'<[^>]+>', '', question_text)
        content = [{"type": "text", "text": str(question_text)}]
        for image_ref in qa_item.question.image_refs.values():
            content.append(self._format_image_message(image_ref))

        return content
    
    def _parse_options(self, qa_item: QAItem) -> List[Any]:
        """Parses the options text and handles placeholders for images."""
        options_content = []
        for opt in qa_item.options:
            options_content.append({"type": "text", "text": f"{opt.id}: "})
            if opt.text is not None:
                options_content.append({"type": "text", "text": str(opt.text)})
            if opt.path :
                options_content.append(self._format_image_message(opt.path))
        return options_content
                
    def _format_prompt_and_messages_for_llm(self, qa_item: QAItem) -> List[Any]:
        """Formats the question, options, and images into messages for the LLM."""
        # Process question text and handle placeholders
        question_content = self._parse_question(qa_item)
        options_content = self._parse_options(qa_item)
        question_content.extend(options_content)
        
        messages = [
            SystemMessage(content="You are an expert VQA assistant. Given a question, associated images, and a list of options (some of which may be images), your task is to choose the best option and respond with its ID only. Do not provide any explanation or any other text. Your answer must be one of the provided option IDs."),
            HumanMessage(content=question_content)
        ]
        
        return messages

    def get_answer(self, qa_item: QAItem) -> str:
        """
        Generates an answer from the LLM for a given question, options, and images.
        The LLM should output one of the provided option IDs.
        """
        messages = self._format_prompt_and_messages_for_llm(qa_item)
        
        valid_option_ids = [option.id for option in qa_item.options]

        if self.mock_mode:
            # In mock mode, just return the first valid option ID
            print(f"Using mock response for question {qa_item.id}")
            time.sleep(0.1)
            return valid_option_ids[0] if valid_option_ids else "mock_response"
        if self.llm == "random":
            # In random mode, just return a random valid option ID
            print(f"Using random response for question {qa_item.id}")
            # time.sleep(0.1)
            return random.choice(valid_option_ids)
            
        try:
            # Only call the actual LLM if not in mock mode
            response = self.llm.invoke(messages)
            model_answer_id = response.content.strip()
            model_answer_id = model_answer_id.lower()

            if model_answer_id not in valid_option_ids:
                found_id = next((opt_id for opt_id in valid_option_ids if opt_id in model_answer_id), None)
                if found_id:
                    model_answer_id = found_id
                else:
                    print(f"No valid ID found in model response (received: {model_answer_id}). Returning unknown answer code.")
                    model_answer_id = random.choice(valid_option_ids)

            
            return model_answer_id
        except Exception as e:
            print(f"Error calling LLM for question {qa_item.id}: {e}")
            return UNKNOWN_ANSWER_CODE


class QADataset:
    """
    Represents a VQA dataset.
    """
    def __init__(self, qa_dataset_path: str):
        self.qa_dataset_path = Path(qa_dataset_path)
        self.dataset: List[QAItem] = []
        self.dataset_by_id = {}
        self.load_dataset()

    def get_item_by_id(self, id: int) -> QAItem:
        """Get a QAItem by its ID."""
        return self.dataset_by_id[id]

    def load_dataset(self, question_range: Optional[List[int]] = None) -> None:
        """Loads the QA dataset from the specified JSON file."""
        if not self.qa_dataset_path.exists():
            print(f"Error: Dataset file not found at {self.qa_dataset_path}")
            self.dataset = []
            return
        
        try:
            with open(self.qa_dataset_path, 'r') as f:
                data = json.load(f)
            self.dataset = [QAItem(**item) for item in data]
            if question_range:
                self.dataset = self.dataset[question_range[0]:question_range[1]]
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.qa_dataset_path}")
            self.dataset = []
        except Exception as e: # Pydantic validation errors will also be caught here
            print(f"Error loading or validating dataset: {e}")
            self.dataset = []

        self.dataset_by_id = {item.id: item for item in self.dataset}
        

class Evaluator:
    """ Given a dataset of questions and answers, evaluate the performance of a model. """

    def __init__(self, qa_dataset: QADataset):
        self.qa_dataset = qa_dataset.dataset
        pass

    def evaluate_from_json(self, model_outputs_path: str) -> None:
        """ Evaluate the performance of a model on the dataset. """
        with open(model_outputs_path, 'r') as f:
            model_outputs = json.load(f)
        self.evaluate(model_outputs)

    def evaluate(self, model_outputs: List[ModelOutput]) -> None:
        """ Evaluate the performance of a model on the dataset. """
        # Calculate accuracies
        # Calculate accuracies
        template_correct_counts: Dict[str, int] = {}
        template_total_counts: Dict[str, int] = {}
        total_correct = 0

        # Need to map question_id back to its template
        question_id_to_template: Dict[int, str] = {item.id: item.template for item in self.qa_dataset}

        for output in model_outputs:
            question_id = output["question_id"]
            template_name = question_id_to_template.get(question_id)
            if template_name:
                template_total_counts[template_name] = template_total_counts.get(template_name, 0) + 1
                if output["correct"] == 1:
                    template_correct_counts[template_name] = template_correct_counts.get(template_name, 0) + 1
                    total_correct += 1
            else:
                print(f"Warning: Could not find template for question ID {question_id}. This question will be excluded from template accuracy calculations.")

        template_accuracies: Dict[str, float] = {}
        for template, total in template_total_counts.items():
            correct = template_correct_counts.get(template, 0)
            template_accuracies[template] = (correct / total) * 100 if total > 0 else 0.0
            template_accuracies[f"{template}_total"] = total
            template_accuracies[f"{template}_correct"] = correct

        num_questions = len(model_outputs)
        
        overall_accuracy = (total_correct / num_questions) * 100 if num_questions > 0 else 0.0

        final_answers_data = template_accuracies.copy()
        final_answers_data["overall_accuracy"] = overall_accuracy
        final_answers_data["num_questions"] = num_questions
        final_answers_data["num_correct"] = total_correct
        return final_answers_data

# --- Evaluation Pipeline ---

class EvaluationPipeline:
    """
    Orchestrates the VQA evaluation process.
    """
    def __init__(self, qa_dataset: QADataset, output_dir: str, llm_client: LLMClient, question_range_str: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.llm_client = llm_client
        self.qa_dataset: List[QAItem] = qa_dataset.dataset
        self.model_outputs: List[ModelOutput] = []
        self.save_freq = 10
        self.model_results_dir = self.output_dir / f"{self.llm_client.model_name}"
        self.model_results_dir.mkdir(parents=True, exist_ok=True)

        # Make filenames unique using question_range_str
        filename_suffix = f"_{question_range_str}" if question_range_str else "_all" # Default to _all if no range
        
        self.model_results_path = self.model_results_dir / f"model_results{filename_suffix}.json"
        self.model_answers_path = self.model_results_dir / f"model_answers{filename_suffix}.json"
        
        # Load existing results if they exist
        self._load_existing_results()

    def _load_existing_results(self) -> None:
        """Load existing results from the last save point if available."""
        if self.model_results_path.exists():
            try:
                with open(self.model_results_path, 'r') as f:
                    existing_results = json.load(f)
                self.model_outputs = [ModelOutput(**result) for result in existing_results]
                self.model_outputs = [
                    output for output in self.model_outputs 
                    if output.answer != UNKNOWN_ANSWER_CODE 
                    and output.question_id in INVALID_GROUPS
                ]
                print(f"Loaded {len(self.model_outputs)} existing results from previous run.")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.model_outputs = []

    def _save_results(self) -> None:
        """Save current results to file."""
        try:
            with open(self.model_results_path, 'w') as f:
                json.dump([output.model_dump() for output in self.model_outputs], f, indent=4)
            print(f"Saved {len(self.model_outputs)} results to {self.model_results_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def run_evaluation(self) -> None:
        """Runs the evaluation loop for each question in the dataset."""
        # Find the last processed question ID
        processed_ids = {output.question_id for output in self.model_outputs \
         if output.answer != UNKNOWN_ANSWER_CODE}
        
        for i, qa_item in enumerate(self.qa_dataset):
            # Skip already processed questions
            if qa_item.id in processed_ids or qa_item.id not in INVALID_GROUPS:
                print(f"Skipping already processed question {qa_item.id}")
                continue

            print(f"Processing question {i+1}/{len(self.qa_dataset)} (ID: {qa_item.id})...")

            model_answer_id = self.llm_client.get_answer(qa_item)
            self._record_and_check_answer(qa_item, model_answer_id, model_reasoning=None)
            
            # Save periodically
            if (i+1) % self.save_freq == 0:
                print(f"Saving metrics at question {i+1}/{len(self.qa_dataset)}...")
                self._save_results()
                self.calculate_and_save_metrics()
        
        # Final save
        print("Saving final results...")
        self._save_results()
        self.calculate_and_save_metrics()
        print("Evaluation run complete.")

    def _record_and_check_answer(self, qa_item: QAItem, model_answer_id: str, model_reasoning: Optional[str] = None) -> None:
        """Records the model's answer and checks if it's correct."""
        is_correct = 1 if model_answer_id == qa_item.answer else 0
        
        # Ensure the model's answer is a valid option ID, even if it was a fallback
        valid_option_ids = [option.id for option in qa_item.options]
        if model_answer_id not in valid_option_ids and model_answer_id != UNKNOWN_ANSWER_CODE:
            model_answer_id = random.choice(valid_option_ids)
            print(f"Warning: Recording answer '{model_answer_id}' for question {qa_item.id} which is not in valid options {valid_option_ids} and not an error code. This might indicate an issue.")
            # Depending on strictness, you might want to mark this as incorrect regardless of qa_item.answer
            is_correct = 1 if model_answer_id == qa_item.answer else 0

        output = ModelOutput(
            model=self.llm_client.model_name,
            answer=model_answer_id,
            question_id=qa_item.id,
            correct=is_correct,
            model_reasoning=model_reasoning
        )
        self.model_outputs.append(output)

    def calculate_and_save_metrics(self) -> None:
        """
        Calculates accuracy per template and overall, then saves all results.
        Saves two files:
        - model_results.json: List of all ModelOutput objects.
        - model_answers.json: Template-wise and overall accuracy.
        """
        if not self.model_outputs:
            print("No model outputs to calculate metrics from.")
            return

        # Calculate accuracies
        template_correct_counts: Dict[str, int] = {}
        template_total_counts: Dict[str, int] = {}
        total_correct = 0

        # Need to map question_id back to its template
        question_id_to_template: Dict[int, str] = {item.id: item.template for item in self.qa_dataset}

        for output in self.model_outputs:
            template_name = question_id_to_template.get(output.question_id)
            if template_name:
                template_total_counts[template_name] = template_total_counts.get(template_name, 0) + 1
                if output.correct == 1:
                    template_correct_counts[template_name] = template_correct_counts.get(template_name, 0) + 1
                    total_correct += 1
            else:
                print(f"Warning: Could not find template for question ID {output.question_id}. This question will be excluded from template accuracy calculations.")

        template_accuracies: Dict[str, float] = {}
        for template, total in template_total_counts.items():
            correct = template_correct_counts.get(template, 0)
            template_accuracies[template] = (correct / total) * 100 if total > 0 else 0.0
            template_accuracies[f"{template}_total"] = total
            template_accuracies[f"{template}_correct"] = correct
        
        overall_accuracy = (total_correct / len(self.model_outputs)) * 100 if len(self.model_outputs) > 0 else 0.0

        final_answers_data = template_accuracies.copy()
        final_answers_data["overall_accuracy"] = overall_accuracy
        final_answers_data["num_questions"] = len(self.model_outputs)
        final_answers_data["num_correct"] = total_correct

        try:
            with open(self.model_answers_path, 'w') as f:
                json.dump(final_answers_data, f, indent=4)
            print(f"Saved model performance metrics to: {self.model_answers_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")


# --- Main Execution ---


def eval_llm(model_name, qa_dataset, output_dir, question_range_str: Optional[str] = None):

    llm_client = LLMClient(model_name=model_name)
    pipeline = EvaluationPipeline(
        qa_dataset=qa_dataset,
        output_dir=output_dir,
        llm_client=llm_client,
        question_range_str=question_range_str
    )

    print(f"Running evaluation with model: {model_name} for range: {question_range_str if question_range_str else 'all'}")
    pipeline.run_evaluation()

    print(f"Calculating and saving metrics to: {output_dir}")
    pipeline.calculate_and_save_metrics()

    print("Evaluation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run VQA evaluation pipeline.")
    parser.add_argument("--qa_dataset_path", 
                        default="data/data_raw/qa.json",
                        type=str,  help="Path to the VQA dataset JSON file.")
    parser.add_argument("--output_dir", 
                        default="data/data_raw/qa/results",
                        type=str, help="Directory to save the evaluation results.")
    parser.add_argument("--model_name", 
                        type=str, default="gemini-2_5-pro-preview-05-06", 
                        help="Comma separated list of model names to use. \
                        E.g. 'gpt-4.1-2025-04-14,claude-3-7-sonnet-20250219'")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (optional, can also be set as env var).")
    parser.add_argument("--question_range", type=str, default="200-400", help="Comma separated list of question ID ranges to evaluate. \
                        E.g. '1-100'")
    parser.add_argument("--mock_mode", type=bool, default=False, help="Whether to run in mock mode.")

    args = parser.parse_args()
    model_names = args.model_name.split(",")
    
    question_range_for_slicing = None
    question_range_for_filename = args.question_range # Use the raw string for filename

    if args.question_range:
        question_range_parts = args.question_range.split("-")
        if len(question_range_parts) == 2:
            try:
                # Convert to 0-indexed for list slicing, adjust end_idx for inclusive behavior if needed by load_dataset
                start_idx = int(question_range_parts[0]) -1 # Assuming 1-indexed input for slicing
                end_idx = int(question_range_parts[1])   # Assuming 1-indexed input for slicing (end is exclusive in Python slices)
                if start_idx < 0: start_idx = 0
                question_range_for_slicing = [start_idx, end_idx]
            except ValueError:
                print(f"Warning: Invalid question_range format: {args.question_range}. Expected START-END. Processing all questions.")
                return 
        else:
            print(f"Warning: Invalid question_range format: {args.question_range}. Expected START-END. Processing all questions.")
            return
    else:
        question_range_for_filename = "all" # Default if no range provided

    if not os.getenv("OPENAI_API_KEY") and not args.api_key:
        print("Error: OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable or provide it with --api_key.")
        return

    try:
        qa_dataset = QADataset(qa_dataset_path=args.qa_dataset_path)
        print(f"Loading dataset from: {args.qa_dataset_path}")
        # Pass the 0-indexed slicing range to load_dataset
        qa_dataset.load_dataset(question_range=question_range_for_slicing) 
        print(f"Found {len(qa_dataset.dataset)} questions to process for range '{question_range_for_filename}'.")

        for model_name in model_names:
            # Pass the string version of question_range for filename construction
            eval_llm(model_name, qa_dataset, args.output_dir, question_range_str=question_range_for_filename) 


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
