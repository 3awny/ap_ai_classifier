"""
Demo script for classifying novel items using LLM approach with full dataset.

This script uses the entire dataset for prompt context retrieval (no splitting)
to classify novel items that may not appear in the training data.
"""
import logging
from typing import Dict, Any, List
from ap_ai_classifier.core import load_and_prepare_data
from ap_ai_classifier.models.llm.pipeline import make_langchain_chain, RetrievalLLMClassifierLC
from ap_ai_classifier.retrieval import SemanticIndex
from ap_ai_classifier.core.preprocessing import normalize_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI client libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def create_novel_item(detail: str, supplier: str = "", net: float = 0.0, vat: float = 0.0) -> Dict[str, Any]:
    """
    Create a novel item dictionary formatted for the LLM classifier.
    
    Args:
        detail: Line item detail text
        supplier: Supplier name (optional)
        net: Net amount (optional, default: 0.0)
        vat: VAT amount (optional, default: 0.0)
        
    Returns:
        Dictionary with model_text and other fields for classification
        Note: Uses lowercase keys (detail, supplier, net, vat) to match prompt template
    """
    # Normalize text
    detail_norm = normalize_text(detail)
    supplier_norm = normalize_text(supplier) if supplier else ""
    
    # Create model_text in the same format as the training data
    if supplier_norm:
        model_text = f"detail: {detail_norm} | supplier: {supplier_norm}"
    else:
        model_text = f"detail: {detail_norm}"
    
    return {
        "detail": detail,  # Lowercase for prompt template
        "supplier": supplier if supplier else "",  # Lowercase for prompt template
        "net": net,  # Lowercase for prompt template
        "vat": vat,  # Lowercase for prompt template
        "model_text": model_text,
        # Keep uppercase versions for display/logging
        "DETAIL": detail,
        "SUPPLIER": supplier,
        "NET": net,
        "VAT": vat,
    }


def classify_novel_items(
    novel_items: List[Dict[str, Any]],
    index: SemanticIndex,
    classifier: RetrievalLLMClassifierLC,
    label_vocab: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Classify a list of novel items using the LLM classifier.
    
    Args:
        novel_items: List of novel item dictionaries
        index: SemanticIndex for retrieval
        classifier: LLM classifier instance
        label_vocab: Label vocabulary dictionary
        
    Returns:
        List of classification results
    """
    results = []
    
    for i, item in enumerate(novel_items, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Classifying Novel Item {i}/{len(novel_items)}")
        logger.info(f"{'='*60}")
        logger.info(f"Detail: {item.get('DETAIL', item.get('detail', ''))}")
        supplier = item.get('SUPPLIER') or item.get('supplier', '')
        if supplier:
            logger.info(f"Supplier: {supplier}")
        
        # Retrieve similar examples for context
        retrieved = index.retrieve(item["model_text"], top_k=5)
        logger.info(f"\n📋 Retrieved {len(retrieved)} similar examples:")
        for j, ex in enumerate(retrieved, 1):
            logger.info(f"  {j}. {ex.text[:80]}...")
            logger.info(f"     → NOMINAL: {ex.nominal}, DEPARTMENT: {ex.department}, TC: {ex.tax_code} (similarity: {ex.similarity:.3f})")
        
        # Classify using LLM
        logger.info(f"\n🤖 Classifying with LLM...")
        prediction = classifier.predict_one(item)
        
        result = {
            "item": item,
            "prediction": prediction,
            "retrieved_examples": retrieved
        }
        results.append(result)
        
        # Display results
        logger.info(f"\n✅ Classification Results:")
        logger.info(f"  NOMINAL: {prediction['nominal']} (confidence: {prediction['confidence']['nominal']:.3f})")
        logger.info(f"  DEPARTMENT: {prediction['department']} (confidence: {prediction['confidence']['department']:.3f})")
        logger.info(f"  TAX CODE: {prediction['tax_code']} (confidence: {prediction['confidence']['tax_code']:.3f})")
        logger.info(f"\n💭 Reasoning:")
        logger.info(f"  {prediction['reasoning']}")
    
    return results


def main():
    """
    Main demo function: classify novel items using LLM approach with full dataset.
    """
    logger.info("\n" + "="*60)
    logger.info("Novel Item Classification Demo")
    logger.info("LLM Approach with Full Dataset Retrieval")
    logger.info("="*60 + "\n")
    
    # Load entire dataset (no splitting)
    logger.info("📂 Loading entire dataset...")
    df, label_vocab = load_and_prepare_data()
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Nominal labels: {len(label_vocab['nominal'])}")
    logger.info(f"   Department labels: {len(label_vocab['department'])}")
    logger.info(f"   Tax code labels: {len(label_vocab['tax_code'])}")
    
    # Prepare data for indexing (use entire dataset)
    logger.info("\n🔍 Building semantic index on entire dataset...")
    texts = df["model_text"].tolist()
    labels = [
        (row["NOMINAL"], row["DEPARTMENT"], row["TC"])
        for _, row in df.iterrows()
    ]
    
    # Build semantic index with OpenAI embeddings
    index = SemanticIndex(backend='openai', use_parallel=False)
    index.fit(texts, labels)
    logger.info(f"   Index built with {len(texts)} items")
    
    # Build LLM classifier chain
    logger.info("\n🔗 Building LLM classifier chain...")
    chain = make_langchain_chain(index, label_vocab)
    classifier = RetrievalLLMClassifierLC(chain)
    logger.info("   Classifier ready!")
    
    # Define novel items to classify
    novel_items = [
        create_novel_item("Coca Cola 300ml"),
        create_novel_item("iPhone 15 plan"),
        create_novel_item("Elevator repair"),
        create_novel_item("Fresh salmon 2kg"),
    ]
    
    logger.info(f"\n📝 Classifying {len(novel_items)} novel items...")
    logger.info("   Items:")
    for i, item in enumerate(novel_items, 1):
        detail = item.get('DETAIL') or item.get('detail', '')
        logger.info(f"   {i}. {detail}")
    
    # Classify novel items
    results = classify_novel_items(novel_items, index, classifier, label_vocab)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Classification Summary")
    logger.info("="*60 + "\n")
    
    for i, result in enumerate(results, 1):
        item = result["item"]
        pred = result["prediction"]
        detail = item.get('DETAIL') or item.get('detail', '')
        logger.info(f"{i}. {detail}")
        logger.info(f"   → NOMINAL: {pred['nominal']}")
        logger.info(f"   → DEPARTMENT: {pred['department']}")
        logger.info(f"   → TAX CODE: {pred['tax_code']}")
        logger.info("")
    
    logger.info("="*60)
    logger.info("Demo completed!")
    logger.info("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()

