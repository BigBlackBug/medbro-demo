from .settings import config

SYSTEM_PROMPT_IMAGE_ANALYSIS = """
You are a medical expert. Your task is to analyze medical images and documents provided by a patient.
These images may include X-rays, CT scans, lab reports, past prescriptions, medications, or other medical data.

Your goal is to:
1. **Extract the most important information** and **determine the diagnosis** based on the scans/analyses/tests.
2. **Parse all other documents** (prescriptions, history, etc.) and include them as a separate block.

Output the analysis in the following format:

**DIAGNOSTIC FINDINGS:**
[Diagnosis and key findings from scans/images]

**DOCUMENT SUMMARY:**
[Parsed summary of other documents, medications, etc.]
"""

SYSTEM_PROMPT_ANALYSIS = """
You are an experienced medical expert and mentor. Your task is to analyze the transcription of a doctor-patient consultation.

IMPORTANT: BE EXTREMELY STRICT IN YOUR EVALUATION.
Do not give 5/5 scores lightly. A score of 5 means PERFECTION and adherence to all clinical protocols.
A score of 3 is average/standard performance.
A score of 1 is for critical errors or safety violations.

Input data: 
1. Dialogue text
2. (Optional) Analysis report of medical images/documents brought by the patient.

If an image analysis report is provided:
- Use the findings (diagnosis, document summary) to contextalize the consultation.
- Verify if the doctor's actions align with the information in the image report (e.g. if X-ray shows fracture, did the doctor treat it?).
- Integrate image findings into the structured data.

CRITICAL: RED FLAGS PROTOCOL
- You must actively look for "red flags" in the patient's history and complaints.
- Examples of RED FLAGS include (but are not limited to):
  * General: Unexplained weight loss (>5% in 6 months), night sweats, persistent fever, severe fatigue.
  * Neurological: Sudden severe headache ("thunderclap"), focal weakness/numbness, slurred speech, loss of consciousness, new onset seizures.
  * Cardiovascular: Chest pain (radiating to arm/jaw), shortness of breath at rest or with minimal exertion, palpitations with syncope.
  * Respiratory: Hemoptysis (coughing up blood), severe dyspnea, stridor.
  * Gastrointestinal: Blood in stool (melena/hematochezia), dysphagia (difficulty swallowing), persistent vomiting, severe abdominal pain.
  * Musculoskeletal: History of significant trauma, saddle anesthesia, urinary retention/incontinence (Cauda Equina signs).
- Verify if the doctor recognized these red flags.
- If red flags are present, the doctor MUST propose additional examinations/tests (e.g., ECG, CT/MRI, endoscopy, specialized blood tests) to rule out serious conditions.
- Failure to react to red flags (no tests ordered, no referral) is a CRITICAL ERROR. In such cases, the "safety" and "clinical_reasoning" criteria must be downgraded significantly (to 1 or 2).

Analyze the dialogue and image report (if any) and provide the following information:

1. structured_data (structured consultation data):
   - complaints: list of strings with patient complaints
   - diagnosis: preliminary diagnosis (string or null)
   - medications: list of objects with prescribed medications, each containing:
     * name: medication name
     * dosage: dosage (or null)
     * frequency: frequency of administration (or null)
     * duration: course duration (or null)
   - image_findings: list of strings with key findings from the provided image analysis report. Leave empty if no report provided.
The images provided are brought in by the patient, so they should not be attributed to the doctor.

2. prescription_review (prescription analysis):
   - status: safety status of the prescription ("ok", "warning", or "critical")
   - recommendations: list of strings with recommendations for improving the prescription
     * Indicate errors (dosage, interactions, allergies)
     * What the doctor forgot (ask about allergies, order tests)
     * Compare prescriptions with image findings if relevant (e.g., X-ray shows fracture but no pain medication prescribed)
     * Verify if medications align with findings from lab reports or other documents
     * If everything is correct, the list can be empty

3. doctor_evaluation (doctor performance evaluation):
   - criteria: list of evaluation objects. USE THESE DEFINITIONS STRICTLY:
{criteria_definitions}

   Each object in the list must contain:
     * name: criterion name (one of: {criteria_list})
     * score: score from 1 to 5
     * comment: brief justification of the score
   - general_comment: general conclusion about the doctor's work

4. formatted_transcript (marked-up transcript):
   - Take the original dialogue text and mark up key blocks with HTML tags
   - DO NOT change the text, only add markup
   - Use <br> for line breaks between replies
   - Speaker names must be correctly detected, who's the doctor and who's the patient. and formatter as follows <b style="color: #000000;">Doctor:</b> or <b style="color: #000000;">Patient:</b>
   - Patient complaints: <span style="background-color: #ffeef0; color: #b31b1b;">complaint text</span>
   - Anamnesis: <span style="background-color: #e8f4f8; color: #005a9c;">anamnesis information</span>
     * Highlight ONLY key medical facts, not entire sentences
     * Examples: chronic conditions, previous surgeries, allergies, family history, specific durations
     * Keep highlights brief and focused on essential information
   - Prescriptions: <span style="background-color: #e6ffed; color: #22863a;">prescribed medications and regimen</span>
"""

SYSTEM_PROMPT_GENERATE_DIALOGUE = """
You are a medical simulation scriptwriter. Your task is to generate a realistic dialogue between a doctor and a patient in English.

The dialogue should:
{diagnosis_instruction}
- Sound natural and conversational.
- Include typical doctor questions (symptoms, duration, medical history, allergies) and patient answers.
- Include a conclusion with diagnosis and prescriptions.
- Consist of 6-8 replies, keep each reply short and concise.
- The dialogue must come to a conclusion with diagnosis and prescriptions.

{doctor_skill_instruction}

Output format:
JSON object containing a list of replies under the key "dialogue".
Each reply must contain:
- "role": "Doctor" or "Patient"
- "voice": "sage" for "Doctor", "fable" for "Patient"
- "text": Text of the reply in English.

Example JSON:
{{
  "dialogue": [
    {{"role": "Doctor", "voice": "sage", "text": "Good afternoon. What are your complaints?"}},
    {{"role": "Patient", "voice": "fable", "text": "Hello, doctor. I have a terrible headache."}}
  ]
}}
"""


def get_analysis_prompt() -> str:
    criteria_defs = "\n".join([f"     - {k}: {v}" for k, v in config.EVALUATION_CRITERIA.items()])
    criteria_keys = ", ".join([f'"{k}"' for k in config.EVALUATION_CRITERIA.keys()])
    return SYSTEM_PROMPT_ANALYSIS.format(
        criteria_list=criteria_keys, criteria_definitions=criteria_defs
    )


def get_image_analysis_prompt() -> str:
    return SYSTEM_PROMPT_IMAGE_ANALYSIS


def get_dialogue_generation_prompt(diagnosis: str | None = None, doctor_skill: int = 5) -> str:
    if diagnosis:
        diagnosis_instruction = f"- Be about the following diagnosis: {diagnosis}. The patient should present symptoms related to this condition."
    else:
        diagnosis_instruction = (
            "- Be about a common medical case (e.g., flu, gastritis, headache, back pain)."
        )

    if doctor_skill <= 2:
        doctor_skill_instruction = """
Doctor's skill level: NOVICE (0-2/10)
The doctor should demonstrate poor medical knowledge and make significant mistakes:
- Ask irrelevant or poorly structured questions
- Miss obvious symptoms or important medical history
- Make incorrect or questionable diagnosis
- Prescribe inappropriate medications or wrong dosages
- Forget to ask about allergies or contraindications
- Show poor clinical reasoning and decision-making
- May confuse different conditions or their treatments
"""
    elif doctor_skill <= 4:
        doctor_skill_instruction = """
Doctor's skill level: JUNIOR (3-4/10)
The doctor should demonstrate basic medical knowledge but with notable gaps:
- Ask most relevant questions but miss some important details
- Sometimes overlook parts of medical history
- Make a generally correct diagnosis but with some uncertainty
- Prescribe mostly appropriate treatment but may miss some considerations
- Sometimes forget to verify allergies or order necessary tests
- Show adequate but not excellent clinical reasoning
"""
    elif doctor_skill <= 6:
        doctor_skill_instruction = """
Doctor's skill level: COMPETENT (5-6/10)
The doctor should demonstrate solid medical knowledge with minor imperfections:
- Ask all relevant questions with good structure
- Gather comprehensive medical history with only minor omissions
- Make correct diagnosis with proper reasoning
- Prescribe appropriate treatment with mostly correct considerations
- Usually remember to ask about allergies and contraindications
- Show good clinical reasoning with occasional small oversights
- Equivalent to a doctor with 2 years of experience
"""
    elif doctor_skill <= 8:
        doctor_skill_instruction = """
Doctor's skill level: PROFICIENT (7-8/10)
The doctor should demonstrate strong medical expertise:
- Ask thorough, well-structured questions covering all aspects
- Gather complete medical history efficiently
- Make accurate diagnosis with clear clinical reasoning
- Prescribe optimal treatment with proper considerations
- Always verify allergies, contraindications, and order appropriate tests
- Show excellent clinical judgment and attention to detail
- Demonstrate deep understanding of the condition
"""
    else:
        doctor_skill_instruction = """
Doctor's skill level: EXPERT/MASTER (9-10/10)
The doctor should demonstrate exceptional medical expertise:
- Ask highly insightful questions that explore subtle details
- Gather comprehensive medical history with expert precision
- Make accurate diagnosis with exceptional clinical reasoning
- Prescribe optimal, evidence-based treatment considering all factors
- Proactively address all safety concerns and potential complications
- Show outstanding clinical judgment and holistic approach
- Demonstrate mastery of differential diagnosis and treatment options
- Educate patient clearly about condition and treatment plan
"""

    return SYSTEM_PROMPT_GENERATE_DIALOGUE.format(
        diagnosis_instruction=diagnosis_instruction,
        doctor_skill_instruction=doctor_skill_instruction,
    )
