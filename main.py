import os
import logging
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from openai import OpenAI

# load environment variables file
load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextProcessor(ABC):
    """Abstract base class for text processing operations."""
    @abstractmethod
    def process(self, text: str, **kwargs) -> str:
        pass


class TextSummarizer(TextProcessor):
    """Handles text summarization using Hugging Face models."""
    def __init__(self, api_key: str, base_url: str = "https://router.huggingface.co/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.available_models = {
            "summarization": "meta-llama/Meta-Llama-3-70B-Instruct"
        }
    
    def process(self, text: str, model_type: str = "summarization", max_length: int = 150) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize
            model_type: Type of model to use for summarization
            max_length: Maximum length of the summary
            
        Returns:
            Generated summary as string
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            if model_type not in self.available_models:
                logger.warning(f"Model type {model_type} not found. Using default summarization model.")
                model_type = "summarization"
            model = self.available_models[model_type]
            summary = self._generate_summary(text, model, max_length)
            logger.info(f"Successfully generated summary using {model}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _generate_summary(self, text: str, model: str, max_length: int) -> str:
        """Generate summary using the specified model."""
        # truncate very long texts to avoid token limits
        truncated_text = self._truncate_text_if_needed(text)
        prompt = self._create_summarization_prompt(truncated_text)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.3  # lower temperature for more factual summaries
            )
        
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _truncate_text_if_needed(self, text: str, max_chars: int = 4000) -> str:
        """Truncate text if it exceeds character limits."""
        if len(text) > max_chars:
            logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
            return text[:max_chars] + "..."
        return text
    
    def _create_summarization_prompt(self, text: str) -> str:
        """Create an effective prompt for text summarization."""
        return f"""
        Please provide a concise and accurate summary of the following text. 
        Focus on the main ideas, key points, and essential information.
        
        TEXT:
        {text}
        
        SUMMARY:
        """


class KeywordExtractor(TextProcessor):
    """Handles keyword extraction from text using Hugging Face models."""
    def __init__(self, api_key: str, base_url: str = "https://router.huggingface.co/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.available_models = {
            "keyword_extraction": "meta-llama/Meta-Llama-3-70B-Instruct"
        }
    
    def process(self, text: str, keyword_count: int = 5, model_type: str = "keyword_extraction") -> str:
        """
        Extract keywords from the input text.
        
        Args:
            text: Input text to analyze
            keyword_count: Number of keywords to extract (1-10)
            model_type: Type of model to use for keyword extraction
            
        Returns:
            Formatted keywords as string
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            # validate keyword count
            if not 1 <= keyword_count <= 10:
                raise ValueError("Keyword count must be between 1 and 10")
            
            if model_type not in self.available_models:
                logger.warning(f"Model type {model_type} not found. Using default keyword extraction model.")
                model_type = "keyword_extraction"
            
            model = self.available_models[model_type]
            keywords = self._extract_keywords(text, keyword_count, model)
            logger.info(f"Successfully extracted {keyword_count} keywords using {model}")
            return keywords
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            return f"Error extracting keywords: {str(e)}"
    
    def _extract_keywords(self, text: str, keyword_count: int, model: str) -> str:
        """Extract keywords using the specified model."""
        truncated_text = self._truncate_text_if_needed(text)
        prompt = self._create_keyword_extraction_prompt(truncated_text, keyword_count)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2  # low temperature for consistent results
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _truncate_text_if_needed(self, text: str, max_chars: int = 3000) -> str:
        """Truncate text if it exceeds character limits."""
        if len(text) > max_chars:
            logger.warning(f"Text truncated from {len(text)} to {max_chars} characters for keyword extraction")
            return text[:max_chars] + "..."
        return text
    
    def _create_keyword_extraction_prompt(self, text: str, keyword_count: int) -> str:
        """Create an effective prompt for keyword extraction."""
        return f"""
        Analyze the following text and extract exactly {keyword_count} most important keywords.
        
        REQUIREMENTS:
        - Select {keyword_count} keywords that best represent the main concepts and topics
        - Keywords should be specific, meaningful, and relevant to the text
        - Focus on nouns, key concepts, and essential terminology
        - Avoid very generic words unless they are central to the text
        - Return the keywords in the exact format specified below
        
        TEXT:
        {text}
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        --- Extracted Keywords ---
        1. [First Keyword]
        2. [Second Keyword]
        3. [Third Keyword]
        ...continue for all {keyword_count} keywords...
        
        Do not include any additional text, explanations, or comments.
        """


class QuizGenerator(TextProcessor):
    """Handles quiz question generation using Hugging Face models."""
    def __init__(self, api_key: str, base_url: str = "https://router.huggingface.co/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.available_models = {
            "quiz_generation": "meta-llama/Meta-Llama-3-70B-Instruct"
        }
    
    def process(self, text: str, question_count: int = 3, model_type: str = "quiz_generation") -> str:
        """
        Generate quiz questions from the input text.
        
        Args:
            text: Input text to analyze
            question_count: Number of questions to generate (1-5)
            model_type: Type of model to use for quiz generation
            
        Returns:
            Formatted quiz questions as string
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            # validate question count
            if not 1 <= question_count <= 5:
                raise ValueError("Question count must be between 1 and 5")
            
            if model_type not in self.available_models:
                logger.warning(f"Model type {model_type} not found. Using default quiz generation model.")
                model_type = "quiz_generation"
            
            model = self.available_models[model_type]
            questions = self._generate_questions(text, question_count, model)
            logger.info(f"Successfully generated {question_count} questions using {model}")
            return questions
            
        except Exception as e:
            logger.error(f"Error in quiz generation: {str(e)}")
            return f"Error generating quiz questions: {str(e)}"
    
    def _generate_questions(self, text: str, question_count: int, model: str) -> str:
        """Generate quiz questions using the specified model."""
        truncated_text = self._truncate_text_if_needed(text)
        prompt = self._create_quiz_generation_prompt(truncated_text, question_count)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _truncate_text_if_needed(self, text: str, max_chars: int = 3500) -> str:
        """Truncate text if it exceeds character limits."""
        if len(text) > max_chars:
            logger.warning(f"Text truncated from {len(text)} to {max_chars} characters for quiz generation")
            return text[:max_chars] + "..."
        return text
    
    def _create_quiz_generation_prompt(self, text: str, question_count: int) -> str:
        """Create an effective prompt for quiz question generation."""
        return f"""
        Based on the following text, create {question_count} multiple-choice questions with 4 answer options each.
        
        REQUIREMENTS:
        - Create exactly {question_count} questions
        - Each question must have 4 answer options (A, B, C, D)
        - Questions should test comprehension of key facts and concepts from the text
        - Make questions varied and cover different aspects of the text
        - Do NOT indicate which answer is correct
        - Ensure all answer options are plausible
        - Base questions only on information present in the text
        
        TEXT:
        {text}
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        ### Question 1
        [Question text?]
        A) [Answer option A]
        B) [Answer option B]
        C) [Answer option C]
        D) [Answer option D]
        
        ### Question 2
        [Question text?]
        A) [Answer option A]
        B) [Answer option B]
        C) [Answer option C]
        D) [Answer option D]
        
        Continue this pattern for all {question_count} questions.
        
        Do not include any additional text, explanations, or mark the correct answers.
        """


class FileHandler:
    """Handles file operations for reading text files."""
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """
        Read text content from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            if not content:
                raise ValueError(f"File is empty: {file_path}")
            logger.info(f"Successfully read file: {file_path} ({len(content)} characters)")
            return content
            
        except UnicodeDecodeError:
            # try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read().strip()
                logger.info(f"File read with latin-1 encoding: {file_path}")
                return content
            except Exception as e:
                logger.error(f"Failed to read file with alternative encoding: {str(e)}")
                raise IOError(f"Cannot read file: {str(e)}")


class UserInputHandler:
    """Handles user input with validation and default values."""
    @staticmethod
    def get_file_path() -> str:
        """
        Get file path from user input with validation.
        
        Returns:
            Valid file path as string
        """
        default_path = "text.txt"
        print(f"\nPlease enter the path to your text file (press Enter for default '{default_path}'):")
        user_input = input("File path: ").strip()
        # use default if user provides empty input
        if not user_input:
            file_path = default_path
            print(f"Using default file path: {file_path}")
        else:
            file_path = user_input
        
        # validate file existence and extension
        return UserInputHandler._validate_file_path(file_path)
    
    @staticmethod
    def get_keyword_count() -> int:
        """
        Get number of keywords to extract from user input with validation.
        Continues prompting until valid input is provided.
        
        Returns:
            Valid keyword count as integer (1-10)
        """
        while True:
            print(f"\nHow many keywords would you like to extract? (1-10):")
            user_input = input("Keyword count: ").strip()
            # check if input is empty
            if not user_input:
                print("Please enter a number between 1 and 10.")
                continue
            
            try:
                keyword_count = int(user_input)
            except ValueError:
                print("Error: Keyword count must be a number. Please try again.")
                continue
            # validate range
            if 1 <= keyword_count <= 10:
                logger.info(f"Keyword count validated: {keyword_count}")
                return keyword_count
            else:
                print("Error: Keyword count must be between 1 and 10. Please try again.")
    
    @staticmethod
    def get_question_count() -> int:
        """
        Get number of quiz questions to generate from user input with validation.
        Continues prompting until valid input is provided.
        
        Returns:
            Valid question count as integer (1-5)
        """
        while True:
            print(f"\nHow many quiz questions would you like to generate? (1-5):")
            user_input = input("Question count: ").strip()
            # check if input is empty
            if not user_input:
                print("Please enter a number between 1 and 5.")
                continue
            
            try:
                question_count = int(user_input)
            except ValueError:
                print("Error: Question count must be a number. Please try again.")
                continue
            # validate range
            if 1 <= question_count <= 5:
                logger.info(f"Question count validated: {question_count}")
                return question_count
            else:
                print("Error: Question count must be between 1 and 5. Please try again.")
    
    @staticmethod
    def _validate_file_path(file_path: str) -> str:
        """
        Validate the provided file path.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated file path
            
        Raises:
            ValueError: If file path is invalid
        """
        # check if file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        # check if it's a file (not a directory)
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        # check file extension
        valid_extensions = {'.txt', '.text', '.md'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in valid_extensions:
            logger.warning(f"File extension '{file_ext}' may not be text-based. Continuing anyway...")
        
        # check file size, prevent reading huge files accidentally
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError(f"File too large ({file_size} bytes). Maximum allowed size is 10MB.")
        
        if file_size == 0:
            raise ValueError("File is empty")
        
        logger.info(f"File path validated: {file_path} ({file_size} bytes)")
        return file_path


class SummaryService:
    """Manages the text summarization process."""
    def __init__(self, api_key: str):
        self.file_handler = FileHandler()
        self.summarizer = TextSummarizer(api_key)
    
    def generate_summary_from_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Generate summary from a text file.
        
        Args:
            file_path: Path to the text file
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # read text from file
            text_content = self.file_handler.read_text_file(file_path)
            # generate summary
            summary = self.summarizer.process(text_content, **kwargs)
            return {
                "success": True,
                "original_text_length": len(text_content),
                "summary": summary,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


class KeywordService:
    """Manages the keyword extraction process."""
    def __init__(self, api_key: str):
        self.file_handler = FileHandler()
        self.keyword_extractor = KeywordExtractor(api_key)
    def extract_keywords_from_file(self, file_path: str, keyword_count: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Extract keywords from a text file.
        
        Args:
            file_path: Path to the text file
            keyword_count: Number of keywords to extract (1-10)
            **kwargs: Additional arguments for keyword extraction
            
        Returns:
            Dictionary containing keywords and metadata
        """
        try:
            # read text from file
            text_content = self.file_handler.read_text_file(file_path)
            # extract keywords
            keywords = self.keyword_extractor.process(text_content, keyword_count=keyword_count, **kwargs)
            
            return {
                "success": True,
                "original_text_length": len(text_content),
                "keywords": keywords,
                "keyword_count": keyword_count,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


class QuizService:
    """Manages the quiz question generation process."""
    def __init__(self, api_key: str):
        self.file_handler = FileHandler()
        self.quiz_generator = QuizGenerator(api_key)
    
    def generate_quiz_from_file(self, file_path: str, question_count: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Generate quiz questions from a text file.
        
        Args:
            file_path: Path to the text file
            question_count: Number of questions to generate (1-5)
            **kwargs: Additional arguments for quiz generation
            
        Returns:
            Dictionary containing quiz questions and metadata
        """
        try:
            # read text from file
            text_content = self.file_handler.read_text_file(file_path)
            # generate quiz questions
            questions = self.quiz_generator.process(text_content, question_count=question_count, **kwargs)
            
            return {
                "success": True,
                "original_text_length": len(text_content),
                "questions": questions,
                "question_count": question_count,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Quiz generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }


def display_welcome_message():
    """Display welcome message and program information."""
    print("=" * 60)
    print("--- AI-Powered Learning Assistant ---")
    print("=" * 60)
    print("This program will:")
    print("1. Read text from a file")
    print("2. Generate a concise summary")
    print("3. Extract key keywords")
    print("4. Create quiz questions")
    print("=" * 60)


def main():
    """Main execution function."""
    # get API key from environment variable
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        logger.error("HUGGINGFACE_API_KEY environment variable not set")
        print("ERROR: Please set HUGGINGFACE_API_KEY environment variable")
        return
    # display welcome message
    display_welcome_message()
    
    # get file path from user with validation
    try:
        file_path = UserInputHandler.get_file_path()
    except ValueError as e:
        print(f"Input error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # generate and display summary
    print("\n" + "=" * 60)
    print("STEP 1: Generating summary...")
    print("=" * 60)
    
    summary_service = SummaryService(api_key)
    print("Generating summary... Please wait.")
    summary_result = summary_service.generate_summary_from_file(
        file_path=file_path,
        model_type="summarization",
        max_length=150
    )
    
    # display summary results
    if summary_result["success"]:
        print("--- TEXT SUMMARY ---")
        print(f"Original text length: {summary_result['original_text_length']} characters")
        print(f"File: {summary_result['file_path']}")
        print("-" * 40)
        print("SUMMARY:")
        print(summary_result["summary"])
    else:
        print(f"SUMMARY ERROR: {summary_result['error']}")
    
    # get keyword count from user with validation
    try:
        keyword_count = UserInputHandler.get_keyword_count()
    except ValueError as e:
        print(f"Input error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # extract and display keywords
    print("\n" + "=" * 60)
    print("STEP 2: Extracting keywords...")
    print("=" * 60)
    
    keyword_service = KeywordService(api_key)
    print("Extracting keywords... Please wait.")
    keyword_result = keyword_service.extract_keywords_from_file(
        file_path=file_path,
        keyword_count=keyword_count,
        model_type="keyword_extraction"
    )
    
    # display keyword results
    if keyword_result["success"]:
        print("--- EXTRACTED KEYWORDS ---")
        print(f"Original text length: {keyword_result['original_text_length']} characters")
        print(f"Requested keywords: {keyword_result['keyword_count']}")
        print(f"File: {keyword_result['file_path']}")
        print("-" * 40)
        print(keyword_result["keywords"])
    else:
        print(f"KEYWORD EXTRACTION ERROR: {keyword_result['error']}")
    
    # get question count from user with validation
    try:
        question_count = UserInputHandler.get_question_count()
    except ValueError as e:
        print(f"Input error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # generate and display quiz questions
    print("\n" + "=" * 60)
    print("STEP 3: Generating quiz questions...")
    print("=" * 60)
    
    quiz_service = QuizService(api_key)
    print("Generating quiz questions... Please wait.")
    quiz_result = quiz_service.generate_quiz_from_file(
        file_path=file_path,
        question_count=question_count,
        model_type="quiz_generation"
    )
    
    # display quiz results
    if quiz_result["success"]:
        print("--- QUIZ QUESTIONS ---")
        print(f"Original text length: {quiz_result['original_text_length']} characters")
        print(f"Generated questions: {quiz_result['question_count']}")
        print(f"File: {quiz_result['file_path']}")
        print("-" * 40)
        print(quiz_result["questions"])
    else:
        print(f"QUIZ GENERATION ERROR: {quiz_result['error']}")
    
    print("\n" + "=" * 60)
    print("Program completed!")
    print("Thank you for using the AI-Powered Learning Assistant!")
    print("=" * 60)


if __name__ == "__main__":
    main()
