from typing import Dict, Any, List, Callable
from pydantic import BaseModel, Field
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, BoundedSemaphore

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import BaseMessage
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    stop_after_attempt,
)

from ap_ai_classifier.models.llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_few_shot_block, parse_llm_response
from ap_ai_classifier.models.llm.metrics import evaluate_llm_classifier
from ap_ai_classifier.retrieval import SemanticIndex
from ap_ai_classifier.config import LLM_MODEL, LLM_TEMPERATURE
from ap_ai_classifier.core import load_and_prepare_data, prepare_multilabel_data, stratified_random_split

logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI client libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _create_llm_with_reasoning(model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
    """Create ChatOpenAI instance with minimal reasoning effort for o-series models."""
    is_reasoning = any(
        prefix in model.lower() 
        for prefix in ['o1', 'o3', 'o4', 'gpt-5']
    )
    
    if is_reasoning:
        # Reasoning models use max_completion_tokens and reasoning effort
        return ChatOpenAI(
            model=model,
            max_completion_tokens=4096,
            reasoning={
                "effort": "minimal"  # Minimal reasoning effort for speed
            },
            max_retries=0  # We handle retries ourselves with custom logic
        )
    else:
        # Standard models use temperature
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            max_retries=0  # We handle retries ourselves with custom logic
        )


class APClassification(BaseModel):
    """Schema for AP line item classification output."""
    
    nominal: str = Field(
        ...,
        description="The nominal / general ledger code classification."
    )
    
    department: str = Field(
        ...,
        description="The department / cost center classification."
    )
    
    tax_code: str = Field(
        ...,
        description="The tax code (TC) classification."
    )
    
    nominal_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the nominal classification (0.0 to 1.0)."
    )
    
    department_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the department classification (0.0 to 1.0)."
    )
    
    tax_code_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the tax code classification (0.0 to 1.0)."
    )
    
    reasoning: str = Field(
        ...,
        description="Concise text explanation (<concise_text_explanation>) describing why this classification was chosen based on the similar examples provided. This reasoning comes from your final prediction output, not from internal thinking."
    )


class LLMClassifierWithRetry:
    """
    LLM classifier wrapper with retry logic and rate limiting.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        schema_class: type[BaseModel],
        max_retries: int = 3,
        max_concurrency: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 20.0
    ):
        """
        Initialize classifier with retry and rate limiting.
        
        Args:
            llm: ChatOpenAI instance
            schema_class: Pydantic BaseModel class for structured output
            max_retries: Maximum retry attempts (default: 3)
            max_concurrency: Maximum concurrent API calls (default: 5)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 20.0)
        """
        self.llm = llm
        self.schema_class = schema_class
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Create semaphore for rate limiting
        self.semaphore = BoundedSemaphore(max_concurrency)
        
        # Apply structured output to schema class
        self.structured_llm = self.llm.with_structured_output(
            schema_class,
            method="json_schema"
        )
        
        # Set up logger for retry attempts
        self.logger = logging.getLogger(__name__)
    
    def _log_retry_attempt(self, retry_state):
        """Log retry attempt before sleeping."""
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number
        self.logger.warning(
            f"LLM API call attempt {attempt_number} failed: {type(exception).__name__}: {exception}. "
            f"Retrying..."
        )
    
    def _invoke_with_retry(self, messages: List[BaseMessage]) -> BaseModel:
        """
        Invoke LLM with tenacity retry decorator for full control over retry behavior and logging.
        
        This wrapper uses tenacity's @retry decorator which supports before_sleep callbacks
        for logging retry attempts, unlike LangChain's with_retry which doesn't expose all parameters.
        
        Args:
            messages: List of BaseMessage objects for the LLM
            
        Returns:
            Structured output instance (schema_class)
            
        Raises:
            RateLimitError: If rate limit exceeded after retries
            APITimeoutError: If API timeout after retries
            APIConnectionError: If connection error after retries
            APIError: If other API errors occur after retries
        """
        # Create a retry wrapper with proper error handling
        retry_decorator = retry(
            retry=retry_if_exception_type((
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                APIError,
            )),
            wait=wait_exponential_jitter(
                initial=self.base_delay,
                max=self.max_delay,
                exp_base=2,
                jitter=0.1
            ),
            stop=stop_after_attempt(self.max_retries + 1),  # +1 because stop_after_attempt counts total attempts
            before_sleep=self._log_retry_attempt  # Log before each retry
        )
        
        @retry_decorator
        def _invoke():
            self.logger.debug("Attempting LLM API call")
            try:
                result = self.structured_llm.invoke(messages)
                self.logger.debug("LLM API call successful")
                return result
            except Exception as e:
                self.logger.debug(f"LLM API call raised exception: {type(e).__name__}: {e}")
                raise
        
        return _invoke()
    
    def _with_rate_limiting(self, func: Callable[[], BaseModel]) -> BaseModel:
        """
        Execute function with semaphore-based rate limiting.
        
        Args:
            func: Function to execute (should be lambda wrapping _invoke_with_retry)
            
        Returns:
            Result from function execution
        """
        # Acquire semaphore (blocks if max_concurrency reached)
        self.semaphore.acquire()
        try:
            return func()
        finally:
            # Always release semaphore
            self.semaphore.release()
    
    def predict(self, messages: List[BaseMessage]) -> BaseModel:
        """
        Predict with structured output, retry logic, and rate limiting.
        
        Args:
            messages: List of BaseMessage objects for the LLM
            
        Returns:
            Structured output instance (schema_class)
        """
        # Execute with rate limiting and retry logic
        result = self._with_rate_limiting(
            lambda: self._invoke_with_retry(messages)
        )
        return result


class RetrievalLLMClassifierLC:
    """
    Thin adapter around the LangChain chain returned by make_langchain_chain().
    
    Important: The reasoning field comes from the structured output (APClassification),
    which is extracted from the LLM's final prediction content, NOT from OpenAI's
    internal reasoning tokens (for reasoning models like o1, o3, o4, gpt-5).
    """
    def __init__(self, chain):
        self.chain = chain

    def predict_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict classification for a single item.
        
        Returns structured output with reasoning extracted from the final prediction JSON,
        NOT from OpenAI's internal reasoning tokens (for reasoning models like o1, o3, o4, gpt-5).
        
        The structured output (with_structured_output) ensures that:
        - Only the final prediction content is parsed according to APClassification schema
        - Internal reasoning tokens are ignored
        - The reasoning field comes from the LLM's explicit reasoning in the structured JSON output
        """
        # Invoke chain - returns APClassification instance (structured output)
        # For reasoning models, this extracts from final content, not reasoning tokens
        result = self.chain.invoke(item)
        
        # Extract reasoning from the structured output (Pydantic model)
        # with_structured_output returns APClassification instance - direct attribute access is standard
        # The schema requires reasoning field, so it should always be present
        reasoning = getattr(result, 'reasoning', '') or ''
        # Ensure it's a string and strip whitespace
        reasoning = str(reasoning).strip() if reasoning else ''
        
        return {
            "nominal": result.nominal,
            "department": result.department,
            "tax_code": result.tax_code,
            "confidence": {
                "nominal": result.nominal_confidence,
                "department": result.department_confidence,
                "tax_code": result.tax_code_confidence,
            },
            "reasoning": reasoning,  # From structured output JSON, NOT from internal reasoning tokens
        }


def predict_batch_threaded(
    texts: List[str],
    index: SemanticIndex,
    llm: ChatOpenAI,
    label_vocab: Dict[str, list],
    max_workers: int = 5
) -> List[Dict[str, str]]:
    """
    Predict labels for a batch of texts using threading for parallel LLM calls.
    
    Args:
        texts: List of text inputs to classify
        index: OpenAI-based retrieval index
        llm: ChatOpenAI instance
        label_vocab: Dictionary of available labels
        max_workers: Number of parallel threads (default: 5)
    
    Returns:
        List of prediction dictionaries with NOMINAL, DEPARTMENT, TC keys
    """
    results = {}
    results_lock = Lock()
    
    def process_sample(idx: int, text: str) -> tuple:
        """Process a single sample and return results."""
        try:
            # Retrieve similar examples
            examples = index.search(text, k=5)
            
            # Format examples
            examples_text = "\n".join([
                f"Example {i+1}: {ex['text'][:100]}...\n  NOMINAL: {ex['labels'][0]}, DEPARTMENT: {ex['labels'][1]}, TC: {ex['labels'][2]}"
                for i, ex in enumerate(examples)
            ])
            
            # Create prompt
            prompt = f"""You are an invoice line item classifier.

Available labels:
NOMINAL: {', '.join(label_vocab['nominal'][:10])}...
DEPARTMENT: {', '.join(label_vocab['department'])}
TC (Tax Code): {', '.join(label_vocab['tax_code'])}

Similar examples from training data:
{examples_text}

Classify this line item:
{text}

Output ONLY these three lines (no explanation):
NOMINAL: <label>
DEPARTMENT: <label>
TC: <label>"""
            
            # Get LLM prediction
            response = llm.invoke(prompt).content
            
            # Parse response using consolidated function
            predictions = parse_llm_response(response)
            
            return idx, predictions
            
        except Exception as e:
            # Return most common labels as fallback
            return idx, {
                'NOMINAL': label_vocab['nominal'][0],
                'DEPARTMENT': label_vocab['department'][0],
                'TC': label_vocab['tax_code'][0]
            }
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_sample, i, text): i 
            for i, text in enumerate(texts)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(texts), desc="Classifying") as pbar:
            for future in as_completed(future_to_idx):
                idx, pred = future.result()
                with results_lock:
                    results[idx] = pred
                pbar.update(1)
    
    # Sort results by original index and return
    return [results[i] for i in range(len(texts))]


def build_langchain_chain(index: SemanticIndex, label_vocab: Dict[str, list]):
    """
    Build a LC chain with OpenAI embeddings for retrieval.
    Uses minimal reasoning effort for reasoning models.
    
    Note: For reasoning models (o1, o3, o4, gpt-5), the reasoning field in APClassification
    comes from the structured output prediction, NOT from OpenAI's internal reasoning tokens.
    The structured output ensures we only extract the reasoning from the final prediction.
    """
    # Create base LLM instance
    llm = _create_llm_with_reasoning(LLM_MODEL, LLM_TEMPERATURE)
    
    # Create classifier with retry and rate limiting
    classifier = LLMClassifierWithRetry(
        llm=llm,
        schema_class=APClassification,
        max_retries=3,
        max_concurrency=5,
        base_delay=1.0,
        max_delay=20.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])

    def add_retrieval(x: Dict[str, Any]) -> Dict[str, Any]:
        retrieved = index.retrieve(x["model_text"])
        few_shot = build_few_shot_block(retrieved)
        x["few_shot_examples"] = few_shot
        x["_retrieved"] = retrieved
        x["nominal_vocab"] = label_vocab["nominal"]
        x["department_vocab"] = label_vocab["department"]
        x["tax_code_vocab"] = label_vocab["tax_code"]
        return x
    
    def invoke_with_classifier(x: Dict[str, Any]) -> APClassification:
        """Invoke classifier with retry and rate limiting."""
        # Format messages for LLM (returns BaseMessage objects)
        formatted_messages = prompt.format_messages(**x)
        # Use classifier with retry logic (structured_llm expects BaseMessage list)
        result = classifier._with_rate_limiting(
            lambda: classifier._invoke_with_retry(formatted_messages)
        )
        return result

    chain = (
        RunnablePassthrough()
        | RunnableLambda(add_retrieval)
        | RunnableLambda(invoke_with_classifier)
    )
    return chain


def evaluate_llm_pipeline():
    """
    Evaluate LLM pipeline with OpenAI embeddings and threading.
    Uses F1-macro score for all labels.
    """
    logger.info("\n" + "="*60)
    logger.info("LLM Pipeline Evaluation with OpenAI Embeddings")
    logger.info("="*60 + "\n")
    
    # Load and prepare data
    logger.info("📂 Loading data...")
    df, label_vocab = load_and_prepare_data()
    logger.info(f"   Total samples: {len(df)}")
    
    # Split data using stratified random split (preserves class distribution and training diversity)
    logger.info("🔮 Creating stratified random split (preserves class distribution)...")
    train_idx, test_idx = stratified_random_split(df, test_size=0.2, random_state=42)
    
    # Prepare multi-label data (prepare_multilabel_data handles pandas Index conversion)
    X_train, X_test, y_train_labels, df_train, df_test = prepare_multilabel_data(
        df, train_idx, test_idx
    )
    
    logger.info(f"\n📦 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build index
    logger.info("\n🔍 Building semantic index with OpenAI embeddings...")
    index = SemanticIndex(backend='openai', use_parallel=False)
    index.fit(X_train, y_train_labels)
    
    # Build LC chain and classifier
    logger.info("\n🔗 Building LangChain pipeline...")
    chain = build_langchain_chain(index, label_vocab)
    clf = RetrievalLLMClassifierLC(chain)
    logger.info("   Pipeline ready!")
    
    # Evaluate using the evaluator module
    logger.info(f"\n⚡ Evaluating on {len(df_test)} samples...")
    logger.info("   (This will make LLM API calls with threading)")
    
    start_time = time.time()
    results = evaluate_llm_classifier(clf, df_test)
    elapsed = time.time() - start_time
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("RESULTS - LLM Pipeline with OpenAI Embeddings")
    logger.info("="*60 + "\n")
    
    logger.info("Accuracy:")
    logger.info(f"  NOMINAL     : {results['nominal_acc']:.4f}")
    logger.info(f"  DEPARTMENT  : {results['department_acc']:.4f}")
    logger.info(f"  TC (Tax Code): {results['tax_code_acc']:.4f}")
    
    logger.info("\nF1-Macro Score:")
    logger.info(f"  NOMINAL     : {results['nominal_f1']:.4f}")
    logger.info(f"  DEPARTMENT  : {results['department_f1']:.4f}")
    logger.info(f"  TC (Tax Code): {results['tax_code_f1']:.4f}")
    
    overall_f1 = (results['nominal_f1'] + results['department_f1'] + results['tax_code_f1']) / 3
    
    logger.info(f"\n🎯 Overall F1-Macro (field-wise average): {overall_f1:.4f}")
    logger.info(f"📊 Sample-wise F1 (accounts for partial correctness): {results['sample_wise_f1']:.4f}")
    logger.info(f"   Time: {elapsed:.1f}s ({elapsed/len(df_test):.2f}s per sample)")
    
    if overall_f1 >= 0.70:
        logger.info(f"   ✅ Target achieved (field-wise)!")
    else:
        logger.warning(f"   ⚠️  Gap to target: {0.70 - overall_f1:.4f}")
    
    # Also check sample-wise F1 against target
    if results['sample_wise_f1'] >= 0.70:
        logger.info(f"   ✅ Target achieved (sample-wise)!")
    else:
        logger.info(f"   📝 Sample-wise F1: {results['sample_wise_f1']:.4f} (shows partial correctness impact)")
    
    # Compare to ML winner
    logger.info("\n" + "="*60)
    logger.info("COMPARISON TO ML APPROACHES")
    logger.info("="*60 + "\n")
    
    logger.info("TC (Tax Code) F1-Macro:")
    logger.info(f"  OpenAI Embeddings + LogReg: 0.8720 ✅ BEST ML")
    logger.info(f"  TF-IDF + BorderlineSMOTE  : 0.8473")
    logger.info(f"  LLM + OpenAI Embeddings   : {results['tax_code_f1']:.4f}")
    
    diff = results['tax_code_f1'] - 0.8720
    if diff > 0:
        logger.info(f"  ✅ LLM is better by {diff:.4f}")
    else:
        logger.warning(f"  ❌ LLM is worse by {abs(diff):.4f}")
    
    # Generate error report CSV
    details_df = results['details']
    # Filter to only rows with errors
    error_df = details_df[
        (details_df['pred_nominal'] != details_df['true_nominal']) |
        (details_df['pred_department'] != details_df['true_department']) |
        (details_df['pred_tc'] != details_df['true_tc'])
    ].copy()
    
    if len(error_df) > 0:
        # Get model name from LLM_MODEL config
        from ap_ai_classifier.config import LLM_MODEL, BASE_DIR
        model_safe_name = LLM_MODEL.replace('.', '-').replace(':', '-')
        error_report_path = BASE_DIR / "artifacts" / f"error_report_{model_safe_name}.csv"
        error_df.to_csv(error_report_path, index=False)
        logger.info(f"\n📊 Error analysis saved to: {error_report_path}")
        logger.info(f"   Found {len(error_df)} misclassified items out of {len(df_test)} total")
    else:
        logger.info(f"\n✅ No errors found! All {len(df_test)} items classified correctly.")
    
    logger.info("\n" + "="*60)
    logger.info(f"Note: Evaluated on full test set ({len(df_test)} samples)")
    logger.info("="*60 + "\n")
    
    return {
        'nominal_f1': results['nominal_f1'],
        'department_f1': results['department_f1'],
        'tax_code_f1': results['tax_code_f1'],
        'overall_f1': overall_f1,
        'sample_wise_f1': results['sample_wise_f1'],  # New metric for partial correctness
        'nominal_acc': results['nominal_acc'],
        'department_acc': results['department_acc'],
        'tax_code_acc': results['tax_code_acc'],
        'time_elapsed': elapsed,
        'details': results['details']
    }


if __name__ == "__main__":
    evaluate_llm_pipeline()
