from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str
    dosage: str | None = None
    frequency: str | None = None
    duration: str | None = None


class StructuredData(BaseModel):
    complaints: list[str] = Field(default_factory=list)
    diagnosis: str | None = None
    medications: list[Medication] = Field(default_factory=list)


class PrescriptionReview(BaseModel):
    status: str  # "ok", "warning", "critical"
    recommendations: list[str] = Field(default_factory=list)


class EvaluationCriterion(BaseModel):
    name: str
    score: int = Field(..., ge=1, le=5)
    comment: str


class DoctorEvaluation(BaseModel):
    criteria: list[EvaluationCriterion]
    general_comment: str


class AnalysisResult(BaseModel):
    structured_data: StructuredData
    prescription_review: PrescriptionReview
    doctor_evaluation: DoctorEvaluation
    formatted_transcript: str
