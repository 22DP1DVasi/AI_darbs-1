# AI-Powered Learning Assistant

This tool provides four basic functions: reading text from a text file, summarizing it into shorter text, extracting keywords from the original text (number of keywords is specified by the user) and generating MSQ quiz questions about the original text (number of questions is specified by the user). All of this is AI-powered, using OpenAI library's client instance and Hugging Face tools to conveniently work with AI models. The main AI model used in this tool is **[meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)**.


This app requires just one environment variable:
- `HUGGINGFACE_API_KEY` - Hugging Face API key
