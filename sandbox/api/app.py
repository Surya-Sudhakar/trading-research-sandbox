from __future__ import annotations
import logging
import re,shutil
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sandbox.research.errors import ResearchError
from sandbox.external_import import ImportError as ExternalImportError
from .models import *
from .service import SandboxReadService

log=logging.getLogger(__name__)
LOCAL_ORIGINS=("http://localhost:5173","http://127.0.0.1:5173")
UPLOAD_CHUNK_BYTES=1024*1024
MAX_INTERPRETATION_REASON_LENGTH=500

def _interpretation_reason(value:str|None):
    if value is None:return None
    value=value.strip()
    if not value:return None
    if len(value)>MAX_INTERPRETATION_REASON_LENGTH:raise HTTPException(422,detail="Interpretation reason exceeds the safe length limit")
    if any(ord(x)<32 and x not in "\t" for x in value) or any(x in value for x in ("<",">","\\","../","..\\")) or re.search(r"(^|\s)([A-Za-z]:[\\/]|/\S)|://",value):raise HTTPException(422,detail="Interpretation reason contains unsafe characters")
    return " ".join(value.split())

def create_app(service:SandboxReadService|None=None):
    app=FastAPI(title="Trading Research Sandbox Read API",version="1.0.0",docs_url=None,redoc_url=None)
    app.state.service=service or SandboxReadService()
    app.add_middleware(CORSMiddleware,allow_origins=list(LOCAL_ORIGINS),allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["Accept","Content-Type"])
    @app.exception_handler(Exception)
    async def safe_error(_request:Request,exc:Exception):
        log.exception("API request failed");return JSONResponse(status_code=500,content=ErrorDTO(error="INTERNAL_ERROR",message="The backend could not complete the safe read request.").model_dump())
    @app.get("/api/system/status",response_model=SystemStatusDTO)
    def system():return app.state.service.system()
    @app.get("/api/strategies",response_model=list[StrategySummaryDTO])
    def strategies():return app.state.service.strategy_summaries()
    @app.get("/api/strategies/{strategy_id}",response_model=StrategyDetailDTO)
    def strategy(strategy_id:str):
        try:return app.state.service.strategy(strategy_id)
        except ResearchError:raise HTTPException(404,detail="Strategy not found")
    @app.get("/api/research/status",response_model=ResearchStatusDTO)
    def research():return app.state.service.research()
    @app.get("/api/data/status",response_model=DataStatusDTO)
    def data():return app.state.service.data()
    @app.get("/api/data/imports/{dataset_id}/diagnostics",response_model=ImportResultDTO)
    def import_diagnostics(dataset_id:str):
        try:return app.state.service.import_diagnostic(dataset_id)
        except ExternalImportError:raise HTTPException(404,detail="Import diagnostics not found")
    @app.post("/api/data/import",response_model=ImportResultDTO,status_code=200)
    async def import_data(file:UploadFile=File(...),provider:ImportProvider=Form(...),symbol:str=Form(...),data_type:str=Form(...),price_type:str=Form(...),source_timezone:str=Form(...),purpose:ImportPurpose=Form(...),timestamp_format:ImportTimestampFormat=Form(ImportTimestampFormat.ISO8601),interpretation_reason:str|None=Form(None)):
        raw_name=file.filename or "";name=Path(raw_name).name
        if not name or name!=raw_name or any(x in raw_name for x in ("/","\\","\x00")) or any(ord(x)<32 for x in raw_name):raise HTTPException(400,detail="Unsafe upload filename")
        lower=name.lower()
        if not lower.endswith((".csv",".csv.gz",".parquet")):raise HTTPException(400,detail="Unsupported file extension")
        symbol=symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9._-]{3,20}",symbol):raise HTTPException(422,detail="Invalid symbol")
        if data_type not in {"TICK","M1","M15"} or price_type not in {"BID","ASK","MID","BID_ASK","UNKNOWN"}:raise HTTPException(422,detail="Invalid import metadata")
        if not source_timezone.strip() or len(source_timezone)>64:raise HTTPException(422,detail="Explicit source timezone is required")
        upload_root=app.state.service.settings.data_dir/"upload_staging";upload_root.mkdir(parents=True,exist_ok=True);size=0
        directory=upload_root/("upload-"+uuid4().hex);directory.mkdir();target=directory/name
        try:
            with target.open("xb") as output:
                while chunk:=await file.read(UPLOAD_CHUNK_BYTES):
                    size+=len(chunk)
                    if size>app.state.service.upload_max_bytes:raise HTTPException(413,detail="Upload exceeds configured size limit")
                    output.write(chunk)
            try:return app.state.service.import_uploaded(target,display_name=name,file_size=size,provider=provider.value,symbol=symbol,data_type=data_type,price_type=price_type,source_timezone=source_timezone.strip(),purpose=purpose.value,timestamp_format=timestamp_format.value,interpretation_reason=_interpretation_reason(interpretation_reason))
            except (ExternalImportError,ValueError,KeyError) as exc:
                log.info("Controlled import rejected: %s",type(exc).__name__);raise HTTPException(400,detail="Import rejected by the Stage 1 validation gateway")
        finally:
            await file.close();shutil.rmtree(directory,ignore_errors=True)
    @app.get("/api/evidence/status",response_model=EvidenceStatusDTO)
    def evidence():return app.state.service.evidence()
    @app.get("/api/audit/status",response_model=AuditStatusDTO)
    def audit():return app.state.service.audit()
    @app.get("/api/market-state/status",response_model=MarketStateStatusDTO)
    def market_state():return app.state.service.market_state()
    return app

app=create_app()
