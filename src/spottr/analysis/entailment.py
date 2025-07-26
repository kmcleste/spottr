import warnings

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer

from spottr.core.models import LogEntry

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class EntailmentScorer:
    """NorGLM-based entailment scoring for log analysis."""

    def __init__(
        self,
        model_id: str = "NorGLM/Entailment",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading entailment model: {model_id}")
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            model_id, fast_tokenizer=True
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model = BertForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        print(f"Entailment model loaded on {self.device}")

    def compute_entailment_scores(
        self, log_texts: list[str], target_statement: str
    ) -> list[float]:
        """
        Compute entailment scores between log texts and target statement.
        Returns list of scores where each score is probability of entailment.
        """
        try:
            if not log_texts:
                return []

            print(
                f"Computing entailment scores for {len(log_texts)} log entries against target: '{target_statement[:50]}...'"
            )

            # Format inputs as: log_text [SEP] target_statement
            input_texts = [
                log_text + " [SEP] " + target_statement for log_text in log_texts
            ]

            # Tokenize inputs
            inputs = self.tokenizer(
                text=input_texts,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            # Create dataset and dataloader
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
            dataloader = DataLoader(dataset, batch_size=self.batch_size)

            all_scores = []
            batch_count = 0

            with torch.no_grad():
                for batch in dataloader:
                    batch_count += 1
                    if batch_count % 10 == 0:
                        print(f"Processing batch {batch_count}/{len(dataloader)}")

                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask = batch

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=None,
                    )
                    logits = outputs.logits

                    # Apply softmax to get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    # Get entailment probabilities (class 1)
                    entailment_probs = probs[:, 1].cpu().numpy()
                    all_scores.extend(entailment_probs)

            print(f"Computed {len(all_scores)} entailment scores")
            return all_scores

        except Exception as e:
            print(f"Error in compute_entailment_scores: {e}")
            import traceback

            traceback.print_exc()
            return []

    def batch_score_insights(
        self, log_entries: list[LogEntry], target_statements: list[str]
    ) -> dict[str, list[float]]:
        """
        Score multiple target statements against log entries.
        Returns dict with target_statement -> scores mapping.
        """
        log_texts = [entry.message for entry in log_entries]
        results = {}

        for target in target_statements:
            scores = self.compute_entailment_scores(log_texts, target)
            results[target] = scores

        return results
