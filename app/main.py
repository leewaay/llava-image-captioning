from argparse import ArgumentParser, RawTextHelpFormatter

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import setup_logging, LoggingMiddleware, llava_parameters, deepl_parameters
from services import llavaImageCaptioning


# Initialize and setup logging.
setup_logging("logs")
logger = structlog.get_logger()


# Define the data model for FastAPI request.
class ImageURL(BaseModel):
    image_url: str


# Create an instance of FastAPI application.
app = FastAPI()

# Add CORS middleware that allows access from all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Integrate the logging middleware.
app.add_middleware(LoggingMiddleware)

# Initialize the LLaVA Image Captioning model.
model = llavaImageCaptioning(deepl_parameters.get("deepl_key"), **llava_parameters)

@app.get("/")
def read_root():
    """Endpoint to check the operational status of the API."""
    return {"message": "API is ready!"}

@app.post("/image_captioning")
def image_captioning(data: ImageURL, request: Request):
    """Endpoint for processing image captioning requests."""
    logger.info("Request received for image_captioning", data=data)
    
    try:
        caption = model.run(data.image_url)
        
        logger.info("Result", response=caption)
        return {"response": caption}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Command-line arguments parser.
def get_args():
    """Parse the command-line arguments for API host and port."""
    parser = ArgumentParser(description='Analysis API for the "LLaVA Image Captioning" project', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-fh', '--host', metavar='host', default="0.0.0.0", help="API Host")
    parser.add_argument('-fp', '--port', metavar='port', type=int, default=5010, help="API Port")
    return parser.parse_args()


# Main execution block.
if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app='__main__:app', host=args.host, port=args.port, log_config=None)