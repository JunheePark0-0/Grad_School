from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

class InputType(BaseModel):
    """입력 형식"""
    input_type : Literal["Query_Only", "Document_Only", "Hybrid"]

class InputQuery(BaseModel):
    """입력 쿼리/문서의 형식"""
    query : str = Field(..., description = "The user's question")
    enhanced_query : str = Field(..., description = "The enhanced, expanded question")
    document_ocr : Optional[dict] = Field(description = "The OCR result of the document")
    input_type : InputType

class InputDocument(BaseModel):
    """입력 문서의 OCR 결과 형식"""
    pass

class DocumentIssue(BaseModel):
    """OCR 결과 문서에서 추출한 쟁점 하나의 형식"""
    issue : str = Field(..., description = "The issue extracted from the document")
    reason : str = Field(..., description = "The reason for the issue")
    risk_summary : str = Field(..., description = "The risk summary of the document")

class RAGOutput(BaseModel):
    """RAG 결과 하나의 형식"""
    text : str = Field(..., description = "The text of the document")
    metadata : dict = Field(..., description = f"{'law_name', 'eff_date', 'law_path', 'section_type', 'junmun_num', 'jomun_num', 'hang_num'}")
    relevance_score : float = Field(..., description = "The relevance score of the document")
    search_rank : int = Field(..., description = "The rank of the document based on relevance")
    source : str = Field(..., description = "The source of the document")

class EnoughContext(BaseModel):
    """외부 검색 필요성 확인(RAG 결과로 충분한 답변 가능 여부)"""
    need_external_search : Literal["YES", "NO"]
    reason : str = Field(..., description = "The reason for the decision (whether or not we need external search)")

class WebSearchOutput(BaseModel):
    """Web Search 결과 하나의 형식"""
    title : str = Field(..., description = "The title of the web page")
    text : str = Field(..., description = "The text of the web page")
    date : Optional[str] = Field(description = "The date of the web page (YYYY-MM-DD)")
    source : str = Field(..., description = "The URL of the web page")
    metadata : dict = Field(..., description = "{'title', 'date', 'editor' etc.}")

class ContextOutput(BaseModel):
    """한 번 더 필터링된 최종 문서 하나의 컨텍스트 형식(RAG + 외부 검색)"""
    doc_type : Literal["Internal_Law", "External_Web"] 
    text : str = Field(..., description = "The text of the document")
    metadata : dict = Field(..., description = "The metadata of the document")
    source : str = Field(..., description = "The source of the document")
    relevance_score : float = Field(..., description = "Re-calculated relevance score of the document")
    rank : int = Field(..., description = "Re-calculated rank of the document based on relevance")

class AnswerOutput(BaseModel):
    """최종 답변 형식"""
    input_type : InputType 
    answer : str = Field(..., description = "The final answer")
    source : List[str] = Field(..., description = "List of sources used when generating the answer")
    context : List[ContextOutput] = Field(..., description = "The total context used when generating the answer")
    risk_summary : str = Field(..., description = "The risk summary for the whole document, when given a document")

class AnswerCorrect(BaseModel):
    """최종 답변 적합성 확인"""
    kind : Literal["CORRECT", "INCORRECT"] 
    feedback : Optional[str] = Field(description = "The feedback based on the context")
