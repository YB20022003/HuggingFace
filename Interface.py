import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import newspaper
from newspaper import Article
import nltk
from nltk.tokenize import sent_tokenize
import re
import time
import langdetect

# Download required NLTK data
nltk.download('punkt')

# Dictionary of available models for summarization with their languages
SUMMARIZATION_MODELS = {
    "facebook/bart-large-cnn": "English",
    "google/pegasus-xsum": "English",
    "facebook/bart-large-xsum": "English",
    "ml6team/mt5-small-german-finetune-mlsum": "German",
    "IlyaGusev/mbart_ru_sum_gazeta": "Russian",
    "csebuetnlp/mT5_multilingual_XLSum": "Multilingual"
}

# Initialize model cache
model_cache = {}

def get_model(model_name):
    """Get or load a model from cache"""
    if model_name not in model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_cache[model_name] = pipeline("summarization", model=model, tokenizer=tokenizer)
    return model_cache[model_name]

def detect_language(text):
    """Detect the language of the text"""
    try:
        return langdetect.detect(text)
    except:
        return "en"  # Default to English if detection fails

def clean_text(text):
    """Clean and preprocess the text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def extract_article(url):
    """Extract article content from a given URL"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text, True
    except Exception as e:
        return None, f"Error extracting article: {str(e)}", False

def process_long_text(text, tokenizer, max_token_length=1024):
    """Process long text by splitting it into manageable chunks"""
    sentences = sent_tokenize(text)
    current_chunk = []
    chunks = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        # Add sentence if it doesn't exceed the max length
        if current_length + len(tokens) <= max_token_length:
            current_chunk.append(sentence)
            current_length += len(tokens)
        else:
            # Add the chunk to chunks list and start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(tokens)
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=150, min_length=40):
    """Summarize the given text using the specified model"""
    if not text or len(text.split()) < 30:
        return "The provided text is too short to summarize effectively."
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Get the model and tokenizer
    summarizer = get_model(model_name)
    tokenizer = summarizer.tokenizer
    
    # Process long text if needed
    if len(tokenizer.tokenize(cleaned_text)) > 1024:
        chunks = process_long_text(cleaned_text, tokenizer)
        summaries = []
        
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length//len(chunks), 
                                min_length=min_length//len(chunks), 
                                do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        final_summary = " ".join(summaries)
    else:
        summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
        final_summary = summary[0]['summary_text']
    
    return final_summary

def process_input(input_type, text_input, url_input, model_name, max_length, min_length):
    """Process either text input or URL based on user selection"""
    start_time = time.time()
    result = {}
    
    if input_type == "Text":
        if not text_input.strip():
            return {"error": "Please enter some text to summarize."}
        
        # Detect language for input validation
        lang = detect_language(text_input)
        model_lang = SUMMARIZATION_MODELS.get(model_name, "English")
        
        # Process the text
        text_to_summarize = text_input
        result["title"] = "User-provided Text"
        
    else:  # URL input
        if not url_input.strip():
            return {"error": "Please enter a valid URL."}
        
        # Extract article from URL
        title, content, success = extract_article(url_input)
        if not success:
            return {"error": content}
        
        text_to_summarize = content
        result["title"] = title
        
        # Detect language for input validation
        lang = detect_language(content)
        model_lang = SUMMARIZATION_MODELS.get(model_name, "English")
    
    # Validate language compatibility with model
    if model_lang != "Multilingual" and lang not in ["en", "de", "ru"] and not (model_lang == "English" and lang == "en") and not (model_lang == "German" and lang == "de") and not (model_lang == "Russian" and lang == "ru"):
        result["warning"] = f"Warning: Detected language ({lang}) may not be compatible with the selected model ({model_name}) which works best with {model_lang}."
    
    # Generate summary
    try:
        summary = summarize_text(text_to_summarize, model_name, max_length, min_length)
        result["summary"] = summary
        result["original_length"] = len(text_to_summarize.split())
        result["summary_length"] = len(summary.split())
        result["processing_time"] = f"{time.time() - start_time:.2f}"
        return result
    except Exception as e:
        return {"error": f"An error occurred during summarization: {str(e)}"}

def format_output(result):
    """Format the output for display"""
    if "error" in result:
        return result["error"]
    
    output = f"Title: {result.get('title', 'N/A')}\n\n"
    
    if "warning" in result:
        output += f"⚠️ {result['warning']}\n\n"
    
    output += f"Summary: {result.get('summary', 'No summary generated.')}\n\n"
    output += f"Original Length: {result.get('original_length', 0)} words\n"
    output += f"Summary Length: {result.get('summary_length', 0)} words\n"
    output += f"Processing Time: {result.get('processing_time', 0)} seconds"
    
    return output

def create_advanced_interface():
    """Create an advanced Gradio interface with more options"""
    with gr.Blocks(title="Advanced News Article Summarizer") as demo:
        gr.Markdown("# Advanced News Article Summarizer")
        gr.Markdown("This application uses Hugging Face's transformers to summarize text or news articles.")
        
        with gr.Tabs():
            with gr.TabItem("Summarize Content"):
                with gr.Row():
                    input_type = gr.Radio(["URL", "Text"], value="URL", label="Input Type")
                
                with gr.Row():
                    # URL input (shown by default)
                    url_input = gr.Textbox(placeholder="Enter the URL of a news article...", 
                                          label="Article URL", 
                                          visible=True)
                    
                    # Text input (hidden by default)
                    text_input = gr.Textbox(placeholder="Enter the text to summarize...", 
                                           label="Text to Summarize", 
                                           lines=10, 
                                           visible=False)
                
                # Update visibility based on input type selection
                input_type.change(
                    fn=lambda x: [x == "URL", x == "Text"],
                    inputs=[input_type],
                    outputs=[url_input, text_input],
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(SUMMARIZATION_MODELS.keys()),
                        value="facebook/bart-large-cnn",
                        label="Summarization Model"
                    )
                
                with gr.Row():
                    with gr.Column():
                        max_length = gr.Slider(
                            minimum=50, maximum=500, value=150, step=10,
                            label="Maximum Summary Length (words)"
                        )
                    with gr.Column():
                        min_length = gr.Slider(
                            minimum=20, maximum=200, value=40, step=10,
                            label="Minimum Summary Length (words)"
                        )
                
                summarize_btn = gr.Button("Summarize")
                
                output_text = gr.Textbox(label="Summary Result", lines=10)
                
                # Set up the processing function
                summarize_btn.click(
                    fn=lambda input_type, text, url, model, max_len, min_len: format_output(
                        process_input(input_type, text, url, model, max_len, min_len)
                    ),
                    inputs=[input_type, text_input, url_input, model_dropdown, max_length, min_length],
                    outputs=[output_text]
                )
                
                # Example inputs
                gr.Examples(
                    examples=[
                        ["URL", "", "https://www.bbc.com/news/world-us-canada-56163220", "facebook/bart-large-cnn", 150, 40],
                        ["URL", "", "https://www.theguardian.com/environment/2022/jul/18/climate-breakdown-human-survival-governments-environment", "google/pegasus-xsum", 200, 50],
                        ["Text", "Artificial intelligence (AI) has become an integral part of our daily lives, from virtual assistants like Siri and Alexa to recommendation systems on streaming platforms and e-commerce websites. The field has seen remarkable progress in recent years, particularly with the advent of deep learning and transformer-based models. Companies like Hugging Face have played a crucial role in this progress by democratizing access to state-of-the-art AI models. Hugging Face started as a company developing a chatbot application but later pivoted to creating open-source tools for natural language processing. Today, it hosts thousands of pre-trained models, datasets, and applications that researchers, developers, and organizations can use to build AI-powered solutions.", "", "facebook/bart-large-xsum", 100, 30]
                    ],
                    inputs=[input_type, text_input, url_input, model_dropdown, max_length, min_length]
                )
            
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About this Application
                
                This News Article Summarizer application uses the power of pre-trained transformer models from Hugging Face to create concise summaries of news articles or any text.
                
                ### Features:
                - Summarize content from a URL or direct text input
                - Multiple summarization models to choose from
                - Adjustable summary length parameters
                - Support for multiple languages (depending on the model)
                
                ### How it works:
                1. When you provide a URL, the application extracts the article content
                2. The text is processed and sent to the selected summarization model
                3. The model generates a concise summary while preserving the key information
                
                ### Models:
                - **facebook/bart-large-cnn**: Optimized for news article summarization in English
                - **google/pegasus-xsum**: Creates very concise, abstractive summaries in English
                - **facebook/bart-large-xsum**: Creates short, to-the-point summaries in English
                - **ml6team/mt5-small-german-finetune-mlsum**: Specialized for German text summarization
                - **IlyaGusev/mbart_ru_sum_gazeta**: Specialized for Russian text summarization
                - **csebuetnlp/mT5_multilingual_XLSum**: Supports summarization in multiple languages
                
                This project was developed as part of the EBC4281-Generative AI in Digital Business and Economics course.
                """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_advanced_interface()
    demo.launch()
