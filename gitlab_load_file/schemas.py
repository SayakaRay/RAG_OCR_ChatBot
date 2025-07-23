from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime

from config import config

########## Login ############
class LoginHistoryEntry(BaseModel):
    timestamp: datetime
    status: str
    reason: str

########## Register ##########
class QueryCredits(BaseModel):
    freePoints: int = config.FREE_QUERY_CREDITS_PER_DAYS * config.POINT_PER_QUERY
    paidPoints: int = 0

class QueryUsage(BaseModel):
    freeUsage: int = 0
    paidUsage: int = 0

class CreateUser(BaseModel):
    user_id: str
    username: str
    password: str
    email: EmailStr
    firstname: str
    lastname: str
    startUpdate: datetime
    lastUpdate: datetime
    uploadUsage: int = 0
    queryUsage: QueryUsage = QueryUsage()
    uploadCredits: int = config.FREE_UPLOAD_CREDITS
    queryCredits: QueryCredits = QueryCredits()
    token: str | None = None
    namespace: str | None = None

class CreateUserAPI(BaseModel):
    user_id: str
    username: str
    password: str
    email: str
    firstname: str
    lastname: str
    startUpdate: datetime
    lastUpdate: datetime
    uploadUsage: int = 0
    queryUsage: QueryUsage = QueryUsage(freeUsage=0, paidUsage=0)
    uploadCredits: int = 0
    queryCredits: QueryCredits = QueryCredits(freePoints=0, paidPoints=0)
    token: str | None = None
    namespace: str = ""

class RegistrationAttempt(BaseModel):
    timestamp: datetime
    status: str
    reason: str

class CreateLogAccount(BaseModel):
    user_id: str
    registered_at: datetime
    deleted_at: datetime | None = None
    login_history: list[LoginHistoryEntry] = []
    registration_attempts: list[RegistrationAttempt]

class UserSystemPrompt(BaseModel):
    user_id: str
    prompt: str = ""

########## Project ##########
class ModelConfig(BaseModel):
    provider: str = "aws"
    model: str
    temperature: float = 0.0
    max_tokens: int

class ModelsConfig(BaseModel):
    qa: ModelConfig = ModelConfig(model="us.amazon.nova-lite-v1:0", max_tokens=2024)
    rephrase: ModelConfig = ModelConfig(model="us.amazon.nova-pro-v1:0", max_tokens=1024)

class ProjectConfig(BaseModel):
    stream: bool = True
    memory: bool = False
    language: str = "Thai"
    k_from_user: int = 3
    th_score_retrieval: float = 0.65
    re_rank: bool = False
    models: ModelsConfig = ModelsConfig()

class CreateProjectRecord(BaseModel):
    owner_id: str
    owner: str
    project_id: str
    project: str
    namespace: str
    user_id: str
    username: str
    role: str = "Owner"
    dislike_count: int = 0
    file: List = []
    userProjectQueryUsage: QueryUsage = QueryUsage()
    chats: List = []
    config: ProjectConfig = ProjectConfig()

class ProjectPrompt(BaseModel):
    namespace: str
    project_id: str
    owner_id: str
    prompt: str =""

class CreateProjectAPIRecord(BaseModel):
    owner_id: str
    owner: str
    project_id: str
    project: str
    namespace: Optional[str]
    user_id: str
    username: str
    role: str = "API"
    project_api: str
    creator_id: str
    createAt: datetime
    dislike_count: int = 0
    file: List = []
    userProjectQueryUsage: QueryUsage = QueryUsage()
    chats: List = []
    config: ProjectConfig = ProjectConfig()

########## file ##########
class MetadataPinecone(BaseModel):
    project_id: str
    filename: str
    page: str
    text: str
    date_upload: str
    upload_by: str

class ProjectUserDetail(BaseModel):
    username: str
    role: str
    display_name: str
    image: str

########### Redeem Code ##########
class CreateRedeemTransactionRecord(BaseModel):
    user_id: str
    code: str
    queryCredits: QueryCredits
    uploadCredits: int
    status: str
    createdAt: datetime

class TokenPayload(BaseModel):
    user_id: str
    username: str
    email: str
    datetime: str