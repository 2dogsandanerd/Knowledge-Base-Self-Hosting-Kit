"""
FastAPI Error Handlers.

Registers global exception handlers for:
- Custom RAG exceptions
- Standard Python exceptions
- FastAPI HTTPException
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from .exceptions import BaseRAGException


async def rag_exception_handler(request: Request, exc: BaseRAGException):
    """
    Handle custom RAG exceptions.

    Logs error and returns structured JSON response.
    """
    logger.error(
        f"RAG Exception: {exc.code.value} - {exc.message}",
        extra={
            "code": exc.code.value,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.

    Converts Pydantic errors to consistent format.
    """
    logger.warning(
        f"Validation Error: {request.url.path}",
        extra={
            "errors": exc.errors(),
            "path": request.url.path
        }
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "message": "Validation failed",
                "code": "INVALID_INPUT",
                "details": {
                    "validation_errors": exc.errors()
                },
                "retryable": False
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.

    Logs full traceback and returns generic error.
    """
    logger.exception(
        f"Unhandled Exception: {type(exc).__name__}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
                "details": {
                    "type": type(exc).__name__
                },
                "retryable": False
            }
        }
    )


def register_error_handlers(app):
    """
    Register all error handlers with FastAPI app.

    Call this in main.py after app creation.
    """
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(BaseRAGException, rag_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers registered")
