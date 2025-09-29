"""Image analysis tool using Ollama VLM."""
import logging
import httpx
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImageAnalysisTool:
    """Tool for analyzing images using Ollama Vision Language Models."""

    def __init__(
        self,
        model: str = "llava",  # Default VLM model in Ollama
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """Initialize the image analysis tool.

        Args:
            model: Ollama VLM model to use (llava, bakllava, etc.)
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

        logger.info(f"Initialized ImageAnalysisTool with model: {model}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image encoding fails
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not image_file.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        # Check if it's likely an image file
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if image_file.suffix.lower() not in valid_extensions:
            logger.warning(f"File {image_path} doesn't have a common image extension")

        try:
            with open(image_file, 'rb') as f:
                image_bytes = f.read()

            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            logger.debug(f"Successfully encoded image: {image_path} ({len(image_bytes)} bytes)")
            return base64_string

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    async def analyze_image(self, image_path: str, question: str) -> str:
        """Analyze an image with a specific question using Ollama VLM.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            Analysis result from the VLM
        """
        logger.info(f"Analyzing image: {image_path}")
        logger.info(f"Question: {question}")

        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)

            # Prepare the request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": question,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }

            logger.debug(f"Sending request to Ollama: {self.base_url}/api/generate")

            # Make API call to Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )

                if response.status_code != 200:
                    error = f"Ollama VLM API error: {response.status_code} - {response.text}"
                    logger.error(error)
                    return f"Error: {error}"

                result = response.json()
                analysis_result = result.get("response", "")

                if not analysis_result:
                    error = "No response from VLM model"
                    logger.error(error)
                    return f"Error: {error}"

                logger.info(f"Successfully analyzed image: {len(analysis_result)} chars response")
                logger.debug(f"VLM response: {analysis_result[:200]}...")

                return analysis_result

        except FileNotFoundError as e:
            error = f"Image file not found: {str(e)}"
            logger.error(error)
            return f"Error: {error}"
        except httpx.TimeoutException:
            error = f"Request timed out after {self.timeout} seconds"
            logger.error(error)
            return f"Error: {error}"
        except Exception as e:
            error = f"Image analysis failed: {str(e)}"
            logger.error(error)
            return f"Error: {error}"

    async def check_model_availability(self) -> bool:
        """Check if the specified VLM model is available in Ollama.

        Returns:
            True if model is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    is_available = self.model in available_models

                    if is_available:
                        logger.info(f"VLM model '{self.model}' is available")
                    else:
                        logger.warning(f"VLM model '{self.model}' not found. Available models: {available_models}")
                        logger.info("You can pull it with: ollama pull llava")

                    return is_available

        except Exception as e:
            logger.error(f"Failed to check VLM model availability: {e}")

        return False

    async def pull_model(self) -> bool:
        """Pull the specified VLM model if not available.

        Returns:
            True if model is ready, False otherwise
        """
        logger.info(f"Pulling VLM model '{self.model}'...")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model, "stream": False}
                )

                if response.status_code == 200:
                    logger.info(f"Successfully pulled VLM model '{self.model}'")
                    return True
                else:
                    logger.error(f"Failed to pull VLM model: {response.text}")

        except Exception as e:
            logger.error(f"Error pulling VLM model: {e}")

        return False

    def __call__(self, image_path: str, question: str) -> str:
        """Synchronous wrapper for analyze_image (for compatibility with sync tools)."""
        import asyncio

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, so we need to run this differently
            import concurrent.futures
            import threading

            def run_in_thread():
                # Create new event loop in thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.analyze_image(image_path, question))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=self.timeout)

        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.analyze_image(image_path, question))


def create_image_analysis_tool(
    model: str = "llava",
    base_url: str = "http://localhost:11434",
    timeout: int = 60
) -> Dict[str, Any]:
    """Create an image analysis tool specification for agents.

    Args:
        model: Ollama VLM model to use
        base_url: Ollama API base URL
        timeout: Request timeout in seconds

    Returns:
        Tool specification with function and metadata
    """
    tool = ImageAnalysisTool(model, base_url, timeout)

    return {
        "name": "image_analysis",
        "description": "Analyze images and answer questions about their content using Vision Language Models. Provide the path to an image file and a question about what you want to know.",
        "function": tool,
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze"
                },
                "question": {
                    "type": "string",
                    "description": "Question about the image content"
                }
            },
            "required": ["image_path", "question"]
        }
    }